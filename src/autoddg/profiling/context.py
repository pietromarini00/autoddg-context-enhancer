from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Literal

from beartype import beartype

from ..llm import LLMClient
from ..utils import load_prompts


# ---------------------------------------------------------------------------
# PDF helpers (module-level so they can be tested/used independently)
# ---------------------------------------------------------------------------


def _extract_text_from_pdf(pdf_path: str) -> str:
    """Extract the full text of a PDF using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted text with pages separated by double newlines.
    """
    import fitz  # PyMuPDF – always available (listed in pyproject.toml)

    doc = fitz.open(pdf_path)
    return "\n\n".join(page.get_text() for page in doc)


def _split_paragraphs(text: str) -> list[str]:
    """Split text into non-empty paragraphs."""
    return [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]


def _find_dataset_mentions(text: str, dataset_title: str) -> list[str]:
    """Return paragraphs that mention the dataset title (case-insensitive)."""
    title_lower = dataset_title.lower()
    title_words = [w for w in title_lower.split() if len(w) > 3]
    threshold = max(1, len(title_words) // 2)
    result: list[str] = []
    for para in _split_paragraphs(text):
        para_lower = para.lower()
        if sum(1 for w in title_words if w in para_lower) >= threshold:
            result.append(para)
    return result


def _find_references_section(text: str) -> str:
    """Return the text from the references/bibliography section onward."""
    text_lower = text.lower()
    match = re.search(
        r"(?:^|\n)[ \t]*(?:references|bibliography|works cited)[ \t]*\n",
        text_lower,
        re.MULTILINE,
    )
    return text[match.start() :] if match else ""


def _find_dataset_in_references(references_text: str, dataset_title: str) -> list[str]:
    """Return reference-list lines that appear to cite the dataset (case-insensitive)."""
    if not references_text:
        return []
    title_words = [w for w in dataset_title.lower().split() if len(w) > 3]
    threshold = max(1, len(title_words) // 2)
    return [
        line.strip()
        for line in references_text.splitlines()
        if line.strip()
        and sum(1 for w in title_words if w in line.lower()) >= threshold
    ]


def _score_chunks_by_keywords(
    text: str,
    keywords: list[str],
    chunk_words: int = 300,
    step_words: int = 150,
    top_k: int = 8,
) -> list[str]:
    """Split *text* into overlapping word-chunks and return the top-k by keyword score."""
    kw_lower = [k.lower() for k in keywords if len(k) > 3]
    words = text.split()
    scored: list[tuple[int, str]] = []
    for i in range(0, len(words), step_words):
        chunk = " ".join(words[i : i + chunk_words])
        score = sum(1 for kw in kw_lower if kw in chunk.lower())
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


@beartype
class ContextFocusedDescription:
    """Extract dataset-specific context from paper PDFs and generate enriched descriptions.

    Five extraction strategies are available.  All can be invoked independently
    via the *method* argument of :meth:`extract_content`:

    ``"docetl"``
        Builds a DocETL map-pipeline over overlapping text chunks from the
        paper.  Each chunk is examined by an LLM for information about the
        target dataset.  Requires the ``docetl`` package.

    ``"reference"``
        Heuristic approach: scans the paper body for paragraphs that mention
        the dataset title and checks the reference list for dataset citations.
        Falls back to prominent section headers when no direct matches exist.
        No LLM involved in the selection step.

    ``"keyword"``
        Ranks overlapping text chunks by how many dataset keywords (title,
        topic, key description terms) they contain.  The top-scoring chunks
        are sent to the LLM which synthesises the enriched description.  No
        LLM in the selection step—faster and cheaper than LLM-based methods.

    ``"llm_selection"``
        Splits the paper into paragraphs and feeds them to the LLM in batches
        together with the dataset title, topic, and description.  The LLM
        decides which paragraphs are specifically about this dataset; the
        selected ones are used to generate the enriched description.

    ``"paragraph_judge"``
        Iterates over every paragraph.  A first keyword filter keeps only
        paragraphs that use data-related vocabulary (data, dataset, corpus,
        benchmark, …).  Each surviving paragraph is then judged by the LLM
        against the dataset description to decide whether it truly describes
        this specific dataset.  All approved paragraphs are concatenated and
        passed to the final description generation step.

    ``"auto"`` (default)
        Cascades through all five methods in order, returning the first
        non-empty result.

    Args:
        client: An :class:`~autoddg.llm.LLMClient` instance or a legacy
            OpenAI-compatible client object.
        model_name: LLM model identifier used for all inference calls.
    """

    def __init__(self, client: LLMClient | Any, model_name: str = "gpt-4o") -> None:
        if isinstance(client, LLMClient):
            self.llm_client = client
        else:
            from ..llm import OpenAICompatibleClient

            self.llm_client = OpenAICompatibleClient(client)
        self.model = model_name
        prompts = load_prompts()["context_profiler"]
        self._system_message = prompts["system_message"].strip()
        self._description_prompt = prompts["description_prompt"]
        self._docetl_chunk_prompt = prompts["docetl_chunk_prompt"]
        self._llm_selection_prompt = prompts["llm_selection_prompt"]
        self._paragraph_judge_prompt = prompts["paragraph_judge_prompt"]

    # ------------------------------------------------------------------
    # Shared LLM call — generates the final enriched description
    # ------------------------------------------------------------------

    def _generate_description_from_context(
        self,
        context_passages: str,
        dataset_title: str | None = None,
        dataset_description: str | None = None,
        dataset_topic: str | None = None,
    ) -> str:
        """Call the LLM to synthesise an enriched description from extracted passages.

        Args:
            dataset_title: Name of the dataset.
            dataset_description: Existing base description.
            context_passages: Relevant text extracted from the paper.
            dataset_topic: Optional short topic string for extra context.

        Returns:
            Enriched description string (may be empty if the model returns nothing).
        """
        topic_line = f"\nDataset topic: {dataset_topic}" if dataset_topic else ""
        prompt = self._description_prompt.format(
            dataset_title=dataset_title,
            dataset_topic=topic_line,
            dataset_description=dataset_description,
            context_passages=context_passages,
        )
        response = self.llm_client.chat_completions_create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_message},
                {"role": "user", "content": prompt},
            ],
        )
        return response["choices"][0]["message"]["content"].strip()

    # ------------------------------------------------------------------
    # Method 1 – DocETL pipeline
    # ------------------------------------------------------------------

    def _extract_via_docetl(
        self,
        pdf_path: str,
        dataset_title: str | None = None,
        dataset_description: str | None = None,
        dataset_topic: str | None = None,
    ) -> str:
        """Use a DocETL map-pipeline to find dataset information across the paper.

        The paper text is split into overlapping word-chunks.  A DocETL
        ``MapOp`` processes each chunk with an LLM prompt asking it to extract
        any information specifically about *dataset_title*.  Non-empty
        extractions are joined and passed to :meth:`_generate_description_from_context`.

        Requires the ``docetl`` package (``pip install docetl``).

        Args:
            pdf_path: Path to the paper PDF.
            dataset_title: Name of the dataset to search for.
            dataset_description: Existing description of the dataset.
            dataset_topic: Optional short topic string.

        Returns:
            Enriched description string, or empty string if nothing found.

        Raises:
            ImportError: If ``docetl`` is not installed.
        """
        from docetl.api import (  # type: ignore[import]
            Dataset,
            MapOp,
            Pipeline,
            PipelineOutput,
            PipelineStep,
        )

        pdf_text = _extract_text_from_pdf(pdf_path)

        chunk_words = 400
        step = chunk_words // 2
        words = pdf_text.split()
        chunks = [
            {"id": idx, "text": " ".join(words[i : i + chunk_words])}
            for idx, i in enumerate(range(0, len(words), step))
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "chunks.json")
            output_path = os.path.join(tmpdir, "output.json")

            with open(input_path, "w", encoding="utf-8") as fh:
                json.dump(chunks, fh)

            topic_line = f"\nDataset topic: {dataset_topic}" if dataset_topic else ""
            chunk_prompt = self._docetl_chunk_prompt.format(
                dataset_title=dataset_title,
                dataset_topic=topic_line,
                dataset_description=dataset_description,
            )

            pipeline = Pipeline(
                name="context_extraction",
                datasets={"chunks": Dataset(type="file", path=input_path)},
                operations=[
                    MapOp(
                        name="extract_dataset_info",
                        type="map",
                        prompt=chunk_prompt + "\n\nText chunk: {{input.text}}",
                        output={"schema": {"dataset_info": "string"}},
                    )
                ],
                steps=[
                    PipelineStep(
                        name="extract_step",
                        input="chunks",
                        operations=["extract_dataset_info"],
                    )
                ],
                output=PipelineOutput(type="file", path=output_path),
                default_model=self.model,
            )

            pipeline.run()

            with open(output_path, encoding="utf-8") as fh:
                results = json.load(fh)

            extracted = [
                r.get("dataset_info", "").strip()
                for r in results
                if r.get("dataset_info", "").strip()
            ]
            if not extracted:
                return ""

            combined = "\n\n".join(extracted)
            return self._generate_description_from_context(
                dataset_title, dataset_description, combined, dataset_topic
            )

    # ------------------------------------------------------------------
    # Method 2 – Heuristic reference / paragraph scanning
    # ------------------------------------------------------------------

    def _extract_via_reference(
        self,
        pdf_path: str,
        dataset_title: str | None = None,
        dataset_description: str | None = None,
        dataset_topic: str | None = None,
    ) -> str:
        """Locate dataset mentions heuristically and synthesise a description.

        Steps:

        1. Find body paragraphs that mention the dataset title.
        2. Find reference-list entries that cite the dataset.
        3. If still empty, extract text under data-related section headers
           (``dataset``, ``methodology``, ``methods``, ``experiments``, …).

        No LLM calls are made for the selection step—only for the final
        description generation.

        Args:
            pdf_path: Path to the paper PDF.
            dataset_title: Name of the dataset to search for.
            dataset_description: Existing description of the dataset.
            dataset_topic: Optional short topic string.

        Returns:
            Enriched description string, or empty string if nothing found.
        """
        text = _extract_text_from_pdf(pdf_path)

        passages = _find_dataset_mentions(text, dataset_title)

        refs_section = _find_references_section(text)
        passages += _find_dataset_in_references(refs_section, dataset_title)

        if not passages:
            header_re = re.compile(
                r"(?:^|\n)[ \t]*"
                r"(?:dataset|data\b|methodology|methods|experimental\s+setup|experiments)"
                r"[ \t]*\n",
                re.IGNORECASE | re.MULTILINE,
            )
            text_lower = text.lower()
            for m in header_re.finditer(text_lower):
                snippet = text[m.start() : m.start() + 2000].strip()
                if snippet:
                    passages.append(snippet)

        if not passages:
            return ""

        combined = "\n\n---\n\n".join(passages[:12])
        #if len(combined) > 9000:
        #    combined = combined[:9000]

        #return self._generate_description_from_context(
        #    dataset_title, dataset_description, combined, dataset_topic
        #)
        return combined

    # ------------------------------------------------------------------
    # Method 3 – Keyword-scored chunk selection (no LLM in selection)
    # ------------------------------------------------------------------

    def _extract_via_keyword(
        self,
        pdf_path: str,
        dataset_title: str | None = None,
        dataset_description: str | None = None,
        dataset_topic: str | None = None,
    ) -> str:
        """Rank text chunks by keyword coverage and send the top ones to the LLM.

        Keywords are drawn from the dataset title, optional topic, and the
        most informative words of the description (words longer than 5 chars).
        The top-scoring chunks are concatenated and sent directly to the LLM
        which synthesises the enriched description.

        This approach requires no LLM calls for the selection step, making it
        faster and cheaper than :meth:`_extract_via_llm_selection` or
        :meth:`_extract_via_paragraph_judge`.  Unlike the DocETL method it
        also requires no external orchestration framework.

        Args:
            pdf_path: Path to the paper PDF.
            dataset_title: Name of the dataset to search for.
            dataset_description: Existing description of the dataset.
            dataset_topic: Optional short topic string.

        Returns:
            Enriched description string, or empty string if nothing found.
        """
        text = _extract_text_from_pdf(pdf_path)

        # Build keyword list from title, topic, and description terms
        keywords: list[str] = dataset_title.split()
        if dataset_topic:
            keywords += dataset_topic.split()
        # Add longer words from the description as extra signal
        if dataset_description:
            keywords += [w for w in dataset_description.split() if len(w) > 5]

        top_chunks = _score_chunks_by_keywords(text, keywords)

        if not top_chunks:
            # Fall back to intro + conclusion when keywords never appear
            words = text.split()
            top_chunks = [
                " ".join(words[:600]),
                " ".join(words[-600:]),
            ]

        context_text = "\n\n---\n\n".join(top_chunks)
        #if len(context_text) > 12000:
        #    context_text = context_text[:12000]

        #return self._generate_description_from_context(
        #    dataset_title, dataset_description, context_text, dataset_topic
        #)
        return context_text
        

    # ------------------------------------------------------------------
    # Method 4 – LLM selects relevant paragraphs (batched)
    # ------------------------------------------------------------------

    def _extract_via_llm_selection(
        self,
        pdf_path: str,
        dataset_title: str | None = None,
        dataset_description: str | None = None,
        dataset_topic: str | None = None,
        batch_size: int = 15,
    ) -> str:
        """Have the LLM decide which paragraphs are about this specific dataset.

        Paragraphs are presented to the LLM in batches together with the
        dataset title, topic, and description.  The LLM returns the relevant
        paragraphs verbatim; the returned text is collected across all batches
        and passed to :meth:`_generate_description_from_context`.

        Args:
            pdf_path: Path to the paper PDF.
            dataset_title: Name of the dataset to search for.
            dataset_description: Existing description of the dataset.
            dataset_topic: Optional short topic string.
            batch_size: Number of paragraphs per LLM call.

        Returns:
            Enriched description string, or empty string if nothing found.
        """
        text = _extract_text_from_pdf(pdf_path)
        paragraphs = _split_paragraphs(text)

        topic_line = f"\nDataset topic: {dataset_topic}" if dataset_topic else ""
        selected: list[str] = []

        for batch_start in range(0, len(paragraphs), batch_size):
            batch = paragraphs[batch_start : batch_start + batch_size]
            prompt = self._llm_selection_prompt.format(
                dataset_title=dataset_title,
                dataset_topic=topic_line,
                dataset_description=dataset_description,
                paragraphs="\n\n".join(batch),
            )
            response = self.llm_client.chat_completions_create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_message},
                    {"role": "user", "content": prompt},
                ],
            )
            content = response["choices"][0]["message"]["content"].strip()
            if content:
                selected.append(content)

        if not selected:
            return ""

        combined = "\n\n---\n\n".join(selected)
        #if len(combined) > 12000:
        #    combined = combined[:12000]

        #return self._generate_description_from_context(
        #    dataset_title, dataset_description, combined, dataset_topic
        #)
        return combined

    # ------------------------------------------------------------------
    # Method 5 – Paragraph-by-paragraph LLM judge
    # ------------------------------------------------------------------

    def _extract_via_paragraph_judge(
        self,
        pdf_path: str,
        dataset_title: str | None = None,
        dataset_description: str | None = None,
        dataset_topic: str | None = None,
    ) -> str:
        """Filter paragraphs with a keyword gate then verify each with an LLM judge.

        Two-stage pipeline:

        1. **Keyword gate** — keeps only paragraphs that contain at least one
           data-related term (``data``, ``dataset``, ``corpus``, ``benchmark``,
           ``collection``, ``table``, ``records``, ``samples``).
        2. **LLM judge** — for each surviving paragraph, asks the LLM whether
           the paragraph specifically discusses the target dataset (yes/no
           decision, informed by the existing dataset description).

        Approved paragraphs are concatenated and used to generate the enriched
        description.

        Args:
            pdf_path: Path to the paper PDF.
            dataset_title: Name of the dataset to search for.
            dataset_description: Existing description of the dataset.
            dataset_topic: Optional short topic string.

        Returns:
            Enriched description string, or empty string if nothing found.
        """
        _DATA_KEYWORDS = {
            "data", "dataset", "corpus", "benchmark", "collection",
            "table", "records", "samples", "observations", "entries",
            "rows", "features", "variables", "instances",
        }

        text = _extract_text_from_pdf(pdf_path)
        paragraphs = _split_paragraphs(text)

        # Stage 1: keyword gate
        candidates = [
            p for p in paragraphs
            if any(kw in p.lower() for kw in _DATA_KEYWORDS)
        ]

        topic_line = f"\nDataset topic: {dataset_topic}" if dataset_topic else ""
        approved: list[str] = []

        for para in candidates:
            prompt = self._paragraph_judge_prompt.format(
                dataset_title=dataset_title,
                dataset_topic=topic_line,
                dataset_description=dataset_description,
                paragraph=para,
            )
            response = self.llm_client.chat_completions_create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_message},
                    {"role": "user", "content": prompt},
                ],
            )
            verdict = response["choices"][0]["message"]["content"].strip().lower()
            if verdict.startswith("yes"):
                approved.append(para)

        if not approved:
            return ""

        combined = "\n\n---\n\n".join(approved)
        #if len(combined) > 12000:
        #    combined = combined[:12000]

        #return self._generate_description_from_context(
        #    dataset_title, dataset_description, combined, dataset_topic
        #)
        return combined

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_content(
        self,
        pdf_path: str,
        dataset_title: str | None = None,
        dataset_description: str | None = None,
        dataset_topic: str | None = None,
        method: Literal[
            "docetl", "reference", "keyword", "llm_selection", "paragraph_judge", "auto"
        ] = "auto",
    ) -> str:
        """Extract paper-derived context for a dataset and return an enriched description.

        Args:
            pdf_path: Filesystem path to the paper PDF.
            dataset_title: Title/name of the dataset to search for in the paper.
            dataset_description: Existing description (used as context for the LLM
                when generating the enriched description and as extra keyword source).
            dataset_topic: Optional short topic string (2-3 words).  Used as an
                additional signal when the dataset title alone is ambiguous.
            method: Which extraction strategy to use:

                * ``"docetl"`` – DocETL map-pipeline over text chunks.
                  Requires ``docetl`` package.
                * ``"reference"`` – Heuristic paragraph + reference scanning.
                  No LLM for selection.
                * ``"keyword"`` – Keyword-scored chunk ranking (title + topic +
                  description terms).  No LLM for selection.
                * ``"llm_selection"`` – LLM picks relevant paragraphs in batches
                  given full dataset context.
                * ``"paragraph_judge"`` – Keyword gate then per-paragraph LLM
                  judge.
                * ``"auto"`` – Tries all methods in the order above, returns
                  first non-empty result (default).

        Returns:
            Additional description derived from the paper, or an empty string
            when no relevant information could be found.

        Raises:
            FileNotFoundError: If *pdf_path* does not exist.
        """
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        kwargs: dict[str, Any] = dict(
            pdf_path=pdf_path,
            dataset_title=dataset_title,
            dataset_description=dataset_description,
            dataset_topic=dataset_topic,
        )

        if method == "docetl":
            return self._extract_via_docetl(**kwargs)
        if method == "reference":
            return self._extract_via_reference(**kwargs)
        if method == "keyword":
            return self._extract_via_keyword(**kwargs)
        if method == "llm_selection":
            return self._extract_via_llm_selection(**kwargs)
        if method == "paragraph_judge":
            return self._extract_via_paragraph_judge(**kwargs)

        # "auto": cascade through all strategies, return first non-empty result
        for approach in (
            self._extract_via_docetl,
            self._extract_via_reference,
            self._extract_via_keyword,
            self._extract_via_llm_selection,
            self._extract_via_paragraph_judge,
        ):
            try:
                result = approach(**kwargs)
                if result:
                    return result
            except Exception:
                continue
        return ""

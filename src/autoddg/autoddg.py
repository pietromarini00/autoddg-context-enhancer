from __future__ import annotations

from typing import Any

from beartype import beartype
from pandas import DataFrame

from .description import DatasetDescriptionGenerator, SearchFocusedDescription
from .evaluation import BaseEvaluator
from .llm import LocalLLMClient, OpenAICompatibleClient
from .profiling import ContextFocusedDescription, SemanticProfiler, profile_dataset
from .topic import DatasetTopicGenerator


@beartype
class AutoDDG:
    """AutoDDG - Automated Dataset Description Generator

    AutoDDG is the library's entry class, exposing:
    * profiling,
    * semantic analysis,
    * topic generation,
    * dataset description (with optional profile, semantic, topic, and paper context),
    * paper-context extraction from research PDFs,
    * search-focused description expansion, and
    * optional description evaluation

    Args:
        client (Any): OpenAI-compatible client (e.g. ``openai.OpenAI(...)``) or None
            for local LLM.
        model_name (str): Model identifier (e.g. ``"gpt-4o"`` for API or
            ``"Qwen/Qwen2.5-7B-Instruct"`` for local).
        use_local_llm (bool): If True, use local LLM via transformers.
            Requires transformers and torch.
        local_llm_device (str | None): Device for local LLM (``"cuda"``, ``"cpu"``,
            or None for auto).
        local_llm_dtype (str | None): Data type for local LLM (``"float16"``,
            ``"bfloat16"``, ``"float32"``, or None for auto).
        description_temperature (float): Sampling temperature for description
            generation.  Lower values produce more deterministic output.
        description_words (int): Target word count for generated descriptions.
        search_model_name (str | None): Override model for search-expansion and
            paper-context extraction.  Defaults to *model_name*.
        semantic_model_name (str | None): Override model for semantic column
            profiling.  Defaults to *model_name*.
        topic_temperature (float): Sampling temperature for topic generation.
        evaluator (BaseEvaluator | None): Optional evaluator for quality scoring.

    Examples:
        Basic usage with API:

            >>> import openai
            >>> from autoddg import AutoDDG
            >>> client = openai.OpenAI(api_key="sk-...")
            >>> pipe = AutoDDG(
            ...     client=client, model_name="gpt-4o", description_words=100
            ... )
            >>> sample_csv = (
            ...     "city,country,population\\nLondon,UK,8908081\\nLeeds,UK,789194"
            ... )
            >>> prompt, desc = pipe.describe_dataset(dataset_sample=sample_csv)
            >>> print(desc)

        Basic usage with local LLM:

            >>> from autoddg import AutoDDG
            >>> pipe = AutoDDG(
            ...     client=None,
            ...     model_name="Qwen/Qwen2.5-7B-Instruct",
            ...     use_local_llm=True,
            ...     description_words=100
            ... )
            >>> sample_csv = (
            ...     "city,country,population\\nLondon,UK,8908081\\nLeeds,UK,789194"
            ... )
            >>> prompt, desc = pipe.describe_dataset(dataset_sample=sample_csv)
            >>> print(desc)

        Advanced usage with topic, paper context, and evaluator:

            The recommended workflow is to first generate a base description from
            the data itself (sample, structural profile, semantic profile, topic),
            then use that description together with the paper PDF to extract
            paper-derived context, and finally regenerate the description including
            that context.

            >>> import pandas as pd
            >>> from autoddg import GPTEvaluator
            >>> df = pd.DataFrame({
            ...     "city": ["London", "Leeds"],
            ...     "country": ["UK", "UK"],
            ...     "population": [8908081, 789194],
            ... })
            >>> profile, semantic = pipe.profile_dataframe(df)
            >>> topic = pipe.generate_topic("UK Cities", None, df.to_csv(index=False))
            >>> # Step 1 — base description from the data alone
            >>> _, base_desc = pipe.describe_dataset(
            ...     dataset_sample=df.to_csv(index=False),
            ...     dataset_profile=profile,
            ...     use_profile=True,
            ...     semantic_profile=semantic,
            ...     use_semantic_profile=True,
            ...     data_topic=topic,
            ...     use_topic=True,
            ... )
            >>> # Step 2 — extract additional context from the paper,
            >>> #           using the base description as contextual signal
            >>> paper_context = pipe.extract_content(
            ...     pdf_path="paper.pdf",
            ...     dataset_title="UK Cities",
            ...     dataset_description=base_desc,
            ...     dataset_topic=topic,
            ... )
            >>> # Step 3 — regenerate description enriched with paper context
            >>> _, desc = pipe.describe_dataset(
            ...     dataset_sample=df.to_csv(index=False),
            ...     dataset_profile=profile,
            ...     use_profile=True,
            ...     semantic_profile=semantic,
            ...     use_semantic_profile=True,
            ...     data_topic=topic,
            ...     use_topic=True,
            ...     data_context=paper_context,
            ...     use_context=True,
            ... )
            >>> evaluator = GPTEvaluator(gpt4_api_key="sk-...")
            >>> pipe.set_evaluator(evaluator)
            >>> scores = pipe.evaluate_description(desc)
            >>> print(scores)
    """

    def __init__(
        self,
        client: Any | None = None,
        model_name: str = "gpt-4o",
        *,
        use_local_llm: bool = False,
        local_llm_device: str | None = None,
        local_llm_dtype: str | None = None,
        description_temperature: float = 0.0,
        description_words: int = 100,
        search_model_name: str | None = None,
        semantic_model_name: str | None = None,
        topic_temperature: float = 0.0,
        evaluator: BaseEvaluator | None = None,
    ) -> None:
        # Initialize LLM client
        if use_local_llm:
            if client is not None:
                raise ValueError(
                    "Cannot specify both client and use_local_llm=True. "
                    "For local LLM, set client=None and use_local_llm=True."
                )
            llm_client = LocalLLMClient(
                model_name=model_name,
                device=local_llm_device,
                torch_dtype=local_llm_dtype,
            )
        elif client is not None:
            llm_client = OpenAICompatibleClient(client)
        else:
            raise ValueError(
                "Must provide either client (for API) or set use_local_llm=True " "(for local LLM)"
            )

        self.client = client  # Keep for backward compatibility
        self.model_name = model_name
        self.description_generator = DatasetDescriptionGenerator(
            client=llm_client,
            model_name=model_name,
            temperature=description_temperature,
            description_words=description_words,
        )
        self.semantic_profiler = SemanticProfiler(
            client=llm_client,
            model_name=semantic_model_name or model_name,
        )
        self.topic_generator = DatasetTopicGenerator(
            client=llm_client,
            model_name=model_name,
            temperature=topic_temperature,
        )
        self.context_extractor = ContextFocusedDescription(
            client=llm_client,
            model_name=search_model_name or model_name,
        )
        self.search_description = SearchFocusedDescription(
            client=llm_client,
            model_name=search_model_name or model_name,
        )
        self.evaluator = evaluator

    def describe_dataset(
        self,
        dataset_sample: str,
        dataset_profile: str | None = None,
        use_profile: bool = False,
        semantic_profile: str | None = None,
        use_semantic_profile: bool = False,
        data_topic: str | None = None,
        use_topic: bool = False,
        data_context: str | None = None,
        use_context: bool = False,
    ) -> tuple[str, str]:
        """
        Produce a dataset description from a CSV sample with optional enrichment signals.

        Each optional signal can be included independently by pairing the data
        argument with its ``use_*`` flag.  For the richest description, pass all
        signals together — including paper-derived context obtained via
        :meth:`extract_content`.

        Args:
            dataset_sample: CSV text containing representative rows of the dataset.
            dataset_profile: Structural profile text produced by
                :meth:`profile_dataframe` (column types, distinct value counts,
                coverage ranges).
            use_profile: Include the structural profile in the prompt if True.
            semantic_profile: Natural-language column semantics produced by
                :meth:`analyze_semantics` (temporal, spatial, entity type, …).
            use_semantic_profile: Include the semantic profile in the prompt if True.
            data_topic: Short 2–3 word topic string produced by
                :meth:`generate_topic`.
            use_topic: Include the topic in the prompt if True.
            data_context: Additional context extracted from an external source
                such as a research paper or codebook.  Typically the string
                returned by :meth:`extract_content`, which contains paper-derived
                information about data collection methodology, research usage,
                known limitations, and other aspects that cannot be inferred from
                the data itself.
            use_context: Include the paper-derived context in the prompt if True.
                When True, the prompt explicitly asks the model to address how the
                data was collected, how it was used in research, and any notable
                characteristics or limitations described in the source.

        Returns:
            (prompt, description) — the full prompt sent to the model and the
            generated description string.
        """

        return self.description_generator.generate_description(
            dataset_sample=dataset_sample,
            dataset_profile=dataset_profile,
            use_profile=use_profile,
            semantic_profile=semantic_profile,
            use_semantic_profile=use_semantic_profile,
            data_topic=data_topic,
            use_topic=use_topic,
            data_context=data_context,
            use_context=use_context,
        )

    def profile_dataframe(self, dataframe: DataFrame) -> tuple[str, str]:
        """
        Summarise structure and coverage using the atlas profiler

        Ref: https://github.com/VIDA-NYU/atlas-profiler#

        Args:
            dataframe: Input frame

        Returns:
            (profile_text, semantic_notes)
        """

        return profile_dataset(dataframe)

    def analyze_semantics(
        self,
        dataframe: DataFrame,
        *,
        use_group_prompting: bool = False,
        use_multi_threading: bool = False,
        use_batch_processing: bool = False,
        max_workers: int | None = None,
        group_size: int = 0,
        batch_size: int = 4,
    ) -> str:
        """
        Infer column semantics with an LLM and return a short overview

        Processing modes available:
        1. Sequential mode (default): Processes columns one by one sequentially.
        2. Multi-threaded mode (use_multi_threading=True): Uses multi-threading to
           process columns in parallel for faster execution. Only available for
           OpenAI API clients.
        3. Group mode (use_group_prompting=True): Processes columns in groups via
           group prompting, reducing API calls.
           - If group_size=0: Processes all columns in a single prompt (most
             efficient).
           - If group_size>0: Processes columns in groups of group_size.
        4. Batch mode (use_batch_processing=True): Processes columns in batches
           using batch inference. Only available for local LLMs. Takes precedence
           over other modes.

        Args:
            dataframe: Input frame
            use_group_prompting: If True, use group prompting (single API call for all
                columns or groups). Takes precedence over use_multi_threading.
            use_multi_threading: If True, use multi-threading for individual column
                processing (only used if use_group_prompting=False and
                use_batch_processing=False). Only works with OpenAI API clients.
            use_batch_processing: If True, use batch processing for local LLMs.
                Only works with LocalLLMClient. Takes precedence over other modes.
            max_workers: Maximum number of concurrent workers for multi-threaded mode.
                Default: min(32, num_columns).
            group_size: Group size for group prompting. If 0, process all columns
                at once. If >0, process in groups of that size.
            batch_size: Batch size for batch processing mode. Only used with local LLMs.

        Returns:
            Summary of column semantics
        """

        return self.semantic_profiler.analyze_dataframe(
            dataframe,
            use_group_prompting=use_group_prompting,
            use_multi_threading=use_multi_threading,
            use_batch_processing=use_batch_processing,
            max_workers=max_workers,
            group_size=group_size,
            batch_size=batch_size,
        )

    def generate_topic(
        self, title: str, original_description: str | None, dataset_sample: str
    ) -> str:
        """
        Generate a 2–3 word topic from title description and sample

        Args:
            title: Dataset title
            original_description: Existing description if available
            dataset_sample: CSV text sample

        Returns:
            Short topic string
        """

        return self.topic_generator.generate_topic(title, original_description, dataset_sample)

    def extract_content(
        self,
        pdf_path: str,
        dataset_title: str | None = None,
        dataset_description: str | None = None,
        dataset_topic: str | None = None,
        method: str = "auto",
    ) -> str:
        """Extract paper-derived context for a dataset and return an enriched description.

        Reads a research paper PDF and produces additional descriptive sentences
        about a specific dataset using information that exists only in the paper —
        such as how the data was collected, how it was used in experiments, and
        what methodology or curation decisions shaped it.

        Delegates to :class:`~autoddg.profiling.ContextFocusedDescription`.

        Args:
            pdf_path: Filesystem path to the paper PDF.
            dataset_title: Title/name of the dataset to search for in the paper.
            dataset_description: Existing description of the dataset.  Used both
                as context for the LLM when generating the enriched description
                and (in keyword-based methods) as an extra source of search terms.
            dataset_topic: Optional short topic string (2–3 words).  Useful when
                the dataset title alone is ambiguous or too generic — the topic
                is included in every LLM prompt to help the model focus on the
                right dataset.
            method: Which extraction strategy to use.  Each strategy differs in
                how it locates the relevant passages inside the paper:

                ``"docetl"``
                    Splits the paper into overlapping text chunks and runs a
                    DocETL map-pipeline over them.  Each chunk is inspected by
                    the LLM, which extracts sentences that specifically describe
                    the target dataset.  Best when the dataset is extensively
                    discussed across many sections.  Requires the ``docetl``
                    package.

                ``"reference"``
                    Heuristic, no LLM in the selection step.  Scans the paper
                    body for paragraphs that contain the dataset title keywords,
                    then checks the reference list for matching entries.  Falls
                    back to prominent section headers (``dataset``,
                    ``methodology``, ``experiments``, …) if no direct title
                    match is found.  Fastest and cheapest option; works best
                    when the dataset is referred to by name throughout the paper.

                ``"keyword"``
                    Also heuristic, no LLM for selection.  Scores every
                    overlapping text chunk by how many keywords from the title,
                    topic, and existing description it contains, then sends the
                    top-scoring chunks to the LLM for description generation.
                    More flexible than ``"reference"`` because it draws keywords
                    from the full description, not just the title.

                ``"llm_selection"``
                    Presents batches of paragraphs to the LLM together with the
                    dataset title, topic, and description, asking it to return
                    the paragraphs that specifically discuss this dataset.
                    Useful when the title alone is not a reliable keyword (e.g.
                    a short or generic name) but the dataset is clearly described
                    in context.  More LLM calls than heuristic methods.

                ``"paragraph_judge"``
                    Two-stage filter over every paragraph in the paper.  First a
                    fast keyword gate keeps only paragraphs that contain
                    data-related vocabulary (``dataset``, ``corpus``,
                    ``benchmark``, ``collected``, …).  Each surviving paragraph
                    is then judged individually by the LLM given the full dataset
                    context.  Most thorough but also most expensive in LLM calls;
                    suitable when the dataset is described implicitly or across
                    many scattered paragraphs.

                ``"auto"`` *(default)*
                    Tries all five strategies in the order above and returns the
                    first non-empty result.  If one strategy raises an exception
                    (e.g. ``docetl`` is not installed) it is silently skipped.

        Returns:
            Additional description derived from the paper, or an empty string
            when no relevant information could be found.
        """
        return self.context_extractor.extract_content(
            pdf_path=pdf_path,
            dataset_title=dataset_title,
            dataset_description=dataset_description,
            dataset_topic=dataset_topic,
            method=method,
        )

    def expand_description_for_search(self, description: str, topic: str) -> tuple[str, str]:
        """
        Expand a readable description into a search-oriented variant

        Args:
            description: Original dataset description
            topic: Topic string

        Returns:
            (prompt, expanded_description)
        """

        return self.search_description.expand_description(description, topic)

    def evaluate_description(self, description: str) -> str:
        """
        Score a description with the configured evaluator

        Args:
            description: Description text to score

        Returns:
            Evaluation response

        Raises:
            RuntimeError: If no evaluator is set
        """

        if self.evaluator is None:
            raise RuntimeError(
                "No evaluator configured for AutoDDG. Provide one via set_evaluator()."
            )
        return self.evaluator.evaluate(description)

    def set_evaluator(self, evaluator: BaseEvaluator) -> None:
        """
        Attach or replace the evaluator to use for scoring

        Args:
            evaluator: Evaluator instance
        """

        self.evaluator = evaluator

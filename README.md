# Contex-Aware Dataset Description

## Overview

This repository contains the official implementation for the **SIGMOD 2026** paper:

> **AutoDDG: Automated Dataset Description Generation using Large Language Models**

AutoDDG is an automated system for generating comprehensive, accurate, readable, and concise dataset descriptions. The framework combines a data-driven approach to summarize dataset contents with large language models (LLMs) to enrich summaries with semantic information and produce human-readable descriptions. AutoDDG supports both API-based (OpenAI) and local LLM (transformers) modes, providing flexibility for different deployment scenarios.

## Installation

Clone the repository and install dependencies via [uv (recommended)](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/VIDA-NYU/AutoDDG.git
cd AutoDDG
uv sync
# If you do not have uv installed:
# * `curl -LsSf https://astral.sh/uv/install.sh | sh`
# * or look at https://docs.astral.sh/uv/getting-started/installation/
```

Then launch Jupyter Lab to explore:

```bash
uv run --with jupyter jupyter lab
``` 

## Getting Started

### Using OpenAI API

The simplest way to use AutoDDG is with an OpenAI API client:

```python
import pandas as pd
from autoddg import AutoDDG, GPTEvaluator
from autoddg.utils import get_sample
from openai import OpenAI

# Setup OpenAI client
my_api_key="sk-..."
client = OpenAI(api_key=my_api_key)
model_name = "gpt-4o-mini"

# Initialize AutoDDG
auto_ddg = AutoDDG(client=client, model_name=model_name)

title = "Ethiopia UEI Survey"
csv_path = "ethiopia_UEI_survey.csv"
pdf_path = "ethiopia_UEI_survey_paper.pdf"
original_description = ''' '''

# Generate description from a small CSV sample
df = pd.read_csv(csv_path, index_col=False, header=1, encoding='latin1')
sample_df, dataset_sample = get_sample(df, sample_size=5)

basic_profile, structural_profile = auto_ddg.profile_dataframe(df)

data_topic = auto_ddg.generate_topic(
    title=title,
    original_description=original_description,
    dataset_sample=dataset_sample,
)

semantic_profile_details = auto_ddg.analyze_semantics(sample_df)

semantic_profile = "\n".join(
    section for section in [structural_profile, semantic_profile_details] if section
)

paper_context = auto_ddg.extract_content(
    pdf_path=pdf_path,
    dataset_title=title,
    dataset_topic=data_topic,
    method={"reference", "keyword", "llm_selection" or "paragraph_judge"}
)

_, original_autoddg_description = auto_ddg.describe_dataset(
    dataset_sample=dataset_sample,
    dataset_profile=basic_profile,
    use_profile=True,
    semantic_profile=semantic_profile,
    use_semantic_profile=True,
    data_topic=data_topic,
    use_topic=True,
)

_, autoddg_enhanced_context_description = auto_ddg.describe_dataset(
    dataset_sample=dataset_sample,
    dataset_profile=basic_profile,
    use_profile=True,
    semantic_profile=semantic_profile,
    use_semantic_profile=True,
    data_topic=data_topic,
    use_topic=True,
    data_context=paper_context,
    use_context=True,
)

## How to Cite

If you use `AutoDDG` in your research, please cite our work:

```bibtex
@misc{2502.01050,
Author = {Haoxiang Zhang and Yurong Liu and Wei-Lun Hung and Aécio Santos and Juliana Freire},
Title = {AutoDDG: Automated Dataset Description Generation using Large Language Models},
Year = {2025},
Eprint = {arXiv:2502.01050},
}
```

---

## License

`AutoDDG` is released under the [Apache License 2.0](./LICENSE).

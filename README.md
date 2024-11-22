<div align="center">
  <img src="docs/images/stencila.png" alt="Stencila" width=300>
</div>
<br>

<p align="center">
  <strong>Stencila Evaluations and Benchmarking</strong>
</p>

<p align="center">
  <a href="#-introduction">
    👋 Intro
  </a> •
  <a href="#-roadmap">
    🚴 Roadmap
  </a> •
  <a href="#%EF%B8%8F-development">
    🛠️ Develop
  </a>
  <a href="#-acknowledgements">
    🙏 Acknowledgements
  </a> •
  <a href="#-supporters">
    💖 Supporters
  </a>
</p>

<div align="center">
  <a href="https://discord.gg/GADr6Jv">
    <img src="https://img.shields.io/discord/709952324356800523.svg?logo=discord&style=for-the-badge&color=1d3bd1&logoColor=66ff66&labelColor=3219a8">
  </a>
</div>
<br>

## 👋 Introduction

Welcome to the repository for Stencila's LLM evaluations and benchmarking. This is in early development and consolidates related code we have had in other repos.

## 🚴 Roadmap

We plan the following three main methodologies to evaluating LLMs for science-focussed prompts and tasks. To avoid discontinuities, we are likely to use a weighting approach, in which we gradually increase the weight of the more advanced methodologies as they are developed.

### Using external benchmarks

Collate external benchmarks and map prompts to each. For example, combine scores from LiveBench's coding benchmark and Aider's code editing benchmark into a single `code-quality` score and use for `stencila/create/code-chunk`, `stencila/create/figure-code` and other code-related prompts.

### Using LLMs-as-a-jury etc

Establish a pipeline for evaluating prompts themselves, and which LLMs are best suited to each prompt, using [LLM-as-a-jury](https://arxiv.org/abs/2404.18796) and other methods for machine-based evaluation.

### Using user acceptance and refinement data

Use data from user's acceptance and refinement of AI suggestions within documents as the basis for human-based evaluations.

## 🛠️ Development

For development, you’ll need to install the following dependencies:

- [uv][uv]
- [just][just]

Then, the following will get you started with a development environment:

```sh
just init
```

Once `uv` is installed, you can use it to install some additional tools:

```sh
uv tool install ruff
uv tool install pyright
```

The `justfile` has some common development-related commands that you might want to run.
For example, the `check` command runs all linting and tests:

```sh
just check
```

To run anything within the virtual environment, you need to use `uv run <command>`.
Alternatively, you can install [direnv], and have the virtual environment activated automatically.
See here for more details about using [direnv and uv][uv-direnv] together.

## 🏃Running the Code

Overview of the current design of the code:

- Code is fetched from the sources defined under the `src/evals/benchmarks` and save the raw data downloaded.
- We then use pydantic classes to validate the incoming data and then save it to parquet data frames using `polars`.
- The `tables` folder contains two tables (as CVS). A set of models with an `id`, and their mapping to the model names in the benchmarks we download. A `use` column lets us pick which models we use. The second table is a list of prompts, with an associated `category`.
- We combine the downloaded parquet data frames with the models and prompts tables to generate lists of scores (validated by pydantic) and then save the results to another scoring data frame.

Each of these stages can be run from the command line.
To see the commands, look in the `pyproject.toml` under the section `[project.script]`. For example, to download the benchmarks, run:
These commands are also invoked from the justfile (`just all`)

### **Under development**

- By default, all the data just gets saved under a `data` folder in the root of the project.
- The scores are currently normalized to 0..1 (rather than 0-100)
- There is no output to any sqlite database yet, thought there is a schema sketched in `src/evals/orm.py`.

## 🙏 Acknowledgements

Thank you to the following projects whose code and/or data we rely on:

- [LiteLLM][litellm]
- [LiveBench][livebench]
- [LMSYS Org][lmsys]
- [TheFastest.ai][fastestai]

## 💖 Supporters

We are grateful for the support of the Astera Institute for this work.

<p align="center"><a href="https://astera.org/"><img src="docs/images/astera.png" height="70"></img></a><p>

[direnv]: https://direnv.net/
[fastestai]: https://thefastest.ai/
[just]: https://github.com/casey/just
[litellm]: https://github.com/BerriAI/litellm
[livebench]: https://livebench.ai/
[lmsys]: https://lmsys.org/
[uv]: https://docs.astral.sh/uv/
[uv-direnv]: https://github.com/direnv/direnv/wiki/Python#uv

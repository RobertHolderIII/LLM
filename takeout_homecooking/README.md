# Takeout to Homecooking

Inspired by [Michelle's](https://www.foodatthecenter.com/) work with helping clients transition from relying on takeout to being able to create their own homecooked meals, this project helps to establish a baseline of one's takeout habits.

## Overview

This approach demonstrates scraping Grubhub history into a Pandas DataFrame, and using an Agno agent to allow basic natural language queries.  This work is adapted from Groq's [Mixture of Agents](https://github.com/groq/groq-api-cookbook/tree/main/tutorials/agno-mixture-of-agents) tutorial.


## Setup

1. Create a virtual environment:
```bash
python -m venv agnoenv
```
2. Activate the virtual environment:
- On Unix or MacOS:
  ```
  source agnoenv/bin/activate
  ```
- On Windows:
  ```
  .\agnoenv\Scripts\activate
  ```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Set up your Groq API key as an environment variable:
```bash
export GROQ_API_KEY=<your-groq-api-key>
```

## Usage

Run the Jupyter notebooks to see various capabilities

- TakeoutAssessment.ipynb - set up agent, tools, and run queries
- Validation-manual.ipynb - assess how well agents are answering the questions


## Requirements

See `requirements.txt` for a full list of dependencies. Key packages include:
- agno
- groq
- pandas

## TODO

- remove repeated code between TakeoutAssessment and Validation-manual
- additional analysis of validation results
- see if an LLM could generate Pandas code to answer queries


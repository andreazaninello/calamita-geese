# The GEESE Challenge for CALAMITA
## Generating and Evaluating Explanations for Semantic Entailment

GEESE (Generating and Evaluating Explanations for Semantic Entailment) is a pipeline to evaluate (generated explanations) for the task of Recognizing Textual Entailment (RTE) in Italian. The challenge focuses on evaluating the impact of generated explanations on the predictive performance of language models. Our methodology assesses the quality of generated explanations by measuring changes in prediction accuracy when explanations are provided.

## Task Description
Consider a pair of sentences $<s_1, s_2>$,  like the ones in the following example:


```text
   (1) Il cielo è grigio oggi.
   
   (2) Faresti bene a prendere l'ombrello.
```

Consider a semantic relation $r$ holding between $s_1$ and $s_2$ (e.g., $s_1$ entails $s_2$, $s_1$ does not entail $s_2$, $s_1$ contradicts $s_2$). Let $E$ be the set of possible explanations for $r$. The task consists in 
- generating an explanation $e_r \in E$ for the semantic relationship $r$ for each $<s_1, s_2>$ in the dataset;
- predict the relation with and without the generated explanation $e_r$;
- assess the quality of the generated explanations $E_{gen}$ by taking the delta between prediction accuracy with and without explanation as a proxy of explanations' quality. 


## GEESE Explanatory Pipeline
Therefore, the GEESE task can be broken down into three subtasks (steps).

### Step 1: Generate Explanation
A first LLM ($M_1$) is prompted to produce explanations  $E_{gen} = \{e_1, e_2, \dots e_n\}$  for a specific semantic relation $r_c$ holding between a given sentence pair, denoted as  $<s_1, s_2>$. In the task, we focus on the entailment relationship, which can take three values: "YES" 
(sentence 1 is entailed by sentence 2), "NO" (sentence 1 is contradicted by sentence 2), "UNKNOWN" (sentence 1 is neither entailed nor contradicted by sentence 2). In our baselines, we focus on one explanation type (why-explanation), but other kinds of explanations or reasoning strategies (like counterfactual or example-based ones) are possible. In our baselines, we use llama-3-3B-instruct \cite{llama3herdmodels} as $M_1$.

### Step 2:  Use Explanation on Relation Prediction
A second LLM ($M_2$) is then provided with the generated explanations $E_{gen}$ to evaluate if the generated explanations improve the task of predicting the correct relations. In practice, this is achieved by appending the explanation as a ``hint'' to the prompt, and ask the model to make a prediction thereof. This process aims to discover how effectively $M_2$ leverages the explanations from $M_1$ to perform the target task. We use llama-3-8B as $M_2$, but other combinations of $M_1$ and $M_2$ are possible.

### Step 3:  Evaluate Explanation Effectiveness 
Explanation effectiveness is evaluated by analyzing how providing different explanations generated in Step 1 affect the model $M_2$ prediction in Step 2. In practice, this is done by calculating the accuracy of the predictions of $M_2$ given the explanations and compare them to the selected baselines. Implement

## Implementation and data
We provide a step-by-step guide to reproduce and extend our experiments in an [interactive notebook](scripts/run_experiments.ipynb). The data produced are contained in the [scripts] folder.

## Baselines
We conduct baseline experiments using Llama-3.1-8B-Instruct as M₁ with a custom implementation in HuggingFace, and Llama-3-8B as M₂, using the LLM-Evaluation-Harness library in a zero-shot setting.

We provide baselines for the following settings:

1. **no-exp**: No explanations provided (baseline);
2. **dummy**: The hypothesis itself (`text_t`) provided as a "non-informative" explanation, controlling for input length and providing a second baseline.
3. **human**: Human-written explanations (from e-RTE-3-it) anonymized (`anon_human`) provided as additional input;
4. **llama-3**: The explanation generated using LLama-3-8B-Instruct as M₁ (`anon_llama3`).

### Example of prompts for zero shots

All experiments have been carried out in a zero-shot setting using the following prompts.

> **M1 - Generation**: `Your task is to provide an explanation for the label assigned for the entailment relationship between two sentences.  
> Sentence 1: **text_t**
> Sentence 2: **text_h**
> Entailment label: **label**.  
> **exp_type**`

> **M2 - Prediction**: `Sentence 1: **text_t**  
> Sentence 2: **text_h**  
> Hint: **anon_explanation**.  
> Entailment label:`

Variables are indicated in **bold**. In prompt M1, `exp_type` = "Explain why." while the other variables are read from each example. In prompt M2, `anon_explanation` can be "Not given." (**no-exp**), = `text_h` (**dummy**), = `anon_human` (**human**), = `anon_llama3` (**llama-3**).

### Baseline Results

Baseline results are reported in the following table:

| **Tasks**     | **n-shot** | **Metric** | **Value** | **Stderr** |
|---------------|------------|------------|-----------|------------|
| geese_dummy   | 0          | acc        | 0.4738    | 0.0177     |
| geese_noexp   | 0          | acc        | 0.4763    | 0.0177     |
| geese_llama3  | 0          | acc        | 0.5425    | 0.0176     |
| geese_human   | 0          | acc        | 0.5787    | 0.0175     |

*Table 1: Results for the 0-shot baseline experiments on the full test set.*




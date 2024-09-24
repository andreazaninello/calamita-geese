# The GEESE Challenge for CALAMITA
## Generating and Evaluating Explanations for Semantic Entailment

GEESE (Generating and Evaluating Explanations for Semantic Entailment) is a pipeline to evaluate (generated explanations) for the task of Recognizing Textual Entailment (RTE) in Italian. The challenge focuses on evaluating the impact of generated explanations on the predictive performance of language models. Our methodology assesses the quality of generated explanations by measuring changes in prediction accuracy when explanations are provided.

## Task Description
Consider a pair of sentences $<s_1, s_2>$,  like the ones in the following example:

```latex
\begin{quote}
\label{quote:working-example}
   (1) \textit{Il cielo è grigio oggi.}
   
   (2) \textit{Faresti bene a prendere l'ombrello.}  
\end{quote}
```

\noindent Consider a semantic relation $r$ holding between $s_1$ and $s_2$ (e.g., $s_1$ entails $s_2$, $s_1$ does not entail $s_2$, $s_1$ contradicts $s_2$). Let $E$ be the set of possible explanations for $r$. The task consists in 
\begin{itemize}
    \item generating an explanation $e_r \in E$ for the semantic relationship $r$ for each $<s_1, s_2>$ in the dataset;
    \item predict the relation with and without the generated explanation $e_r$;
    \item assess the quality of the generated explanations $E_{gen}$ by taking the delta between prediction accuracy with and without explanation as a proxy of explanations' quality. 
\end{itemize}

## GEESE Explanatory Pipeline
The GEESE task can be broken down into three subtasks (steps).

### Step 1: Generate Explanation
A first LLM ($M_1$) is prompted to produce explanations  $E_{gen} = \{e_1, e_2, \dots e_n\}$  for a specific semantic relation $r_c$ holding between a given sentence pair, denoted as  $<s_1, s_2>$. In the task, we focus on the entailment relationship, which can take three values: "YES" 
(sentence 1 is entailed by sentence 2), "NO" (sentence 1 is contradicted by sentence 2), "UNKNOWN" (sentence 1 is neither entailed nor contradicted by sentence 2). In our baselines, we focus on one explanation type (why-explanation), but other kinds of explanations or reasoning strategies (like counterfactual or example-based ones) are possible. In our baselines, we use llama-3-3B-instruct \cite{llama3herdmodels} as $M_1$.

### Step 2:  Use Explanation on Relation Prediction
A second LLM ($M_2$) is then provided with the generated explanations $E_{gen}$ to evaluate if the generated explanations improve the task of predicting the correct relations. In practice, this is achieved by appending the explanation as a ``hint'' to the prompt, and ask the model to make a prediction thereof. This process aims to discover how effectively $M_2$ leverages the explanations from $M_1$ to perform the target task. We use llama-3-8B as $M_2$, but other combinations of $M_1$ and $M_2$ are possible.

### Step 3:  Evaluate Explanation Effectiveness 
Explanation effectiveness is evaluated by analyzing how providing different explanations generated in Step 1 affect the model $M_2$ prediction in Step 2. In practice, this is done by calculating the accuracy of the predictions of $M_2$ given the explanations and compare them to the selected baselines.



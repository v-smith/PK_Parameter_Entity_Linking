from collections import defaultdict
from typing import List

import numpy as np
import openai
from datasets import tqdm
from transformers.integrations import tiktoken

from pk_el.evaluation import evaluate, plot_confusion_matrix

SYSTEM_PROMPT_COT = """
You are a highly intelligent and accurate pharmacokinetics (PK) entity linker. 
You will receive a parameter mention from scientific literature and an pk_ontology. 
Your task is to accurately identify the mention using the concepts in the pk_ontology.

Please answer the following questions to determine the output:

Q1. Does the mention refer to a PK parameter? 
This excludes the following: pharmacodynamic, PBPK, chemical, in vitro, or clinical concepts.
- If yes, go to Q2. 
- If no, set the final answer to NIL. 
- If you are unsure, set the final answer to NIL.

Q2. Does the mention match or refer to a known PK concept in the pk_ontology?
Note mentions may include prefixes, suffixes, time annotations, or drug-specific subscripts.
Map these to their core pk_ontology concept if applicable. 
- If yes, return the concept name exactly as it appears in the pk_ontology.
- If no, set the final answer to NIL. 
- If you are unsure, set the final answer to NIL.

Please return only the final answer in the format: {param: final answer}.
"""

SYSTEM_PROMPT_STANDARD = """
You are a highly intelligent and accurate pharmacokinetics (PK) entity linker. 
You will receive a parameter mention from scientific literature and an pk_ontology. 
Your task is to accurately identify the mention using the concepts in the given pk_ontology.

- Accuracy is paramount. If the text does not precisely refer to any of the concepts listed below, please answer NIL.
- If the mention does not refer to a PK parameter (e.g. part of a pharmacodynamic parameter (e.g. "[MENTION]AUC[\MENTION]/MIC"), or a PBPK, chemical, in vitro, clinical, or other unrelated term), please answer NIL.
- If you are unsure, please answer NIL.
- If there is a match, please answer with the corresponding concept name (PARAM_NAME), exactly as it appears in the pk_ontology below. 

Please return your answer in the format: {param: <answer>}.
"""



SENTENCE_EXAMPLES = """
The following examples are provided to guide you. They include mentions (with context if relevant), expected answers and explanations: 

[Mention] plasma concentration (or its ratio)
[Expected Output] {param: NIL}
[Explanation] Non-specific or ratio-based references to concentration without a specific PK parameter should link to NIL.

[Mention] Maximum concentration at steady-state (Css (max))
[Expected Output] {param: NIL}
[Explanation] This is a mixture of parameters in our pk_ontology (Css, Cmax) and does not fit exactly with any one so should be marked as NIL. 

[Mention] Rate constant for distribution to the effector compartment
[Expected Output] {param: NIL}
[Explanation] Although it refers to a rate constant, it is related to the effect compartment, not specifically covered in our pk_ontology.

[Mention] half-lives[/MENTION] were 1.39 hours and 1.89 hours for R-BSO and S-BSO, respectively.
[Expected Output] {param: t1/2z}
[Explanation]: Unqualified or overall half-lives default to t1/2z.

[Mentions] A pharmacokinetic model with two compartments described the [MENTION]elimination half-lives[/MENTION] of cefepime as 1.65 h in foals and 1.09 h in dogs.
The [MENTION]half-life of the later phase[/MENTION] was 323 minutes in healthy individuals, with comparable values in cholestasis and biliary obstruction groups.
[Expected Outputs in both cases] {param: t1/2β}
[Explanation]: Later phase-specific half-lives should map to t1/2β. Remember in the context of a 2-compartment model, terminal/elimination t1/2 refer to t1/2β and in a three-compartment model these refer to t1/2γ.   

[Mention] The [MENTION]half-life of the fast distribution phase[/MENTION] was estimated.
[Expected Output] {param: t1/2α}
[Explanation] Early phase-specific half-lives should map to t1/2α.

[Mentions] dialytic clearance, CL(uptake), clearance from the perfusate (CL) and into the bile (CLB)
[Expected Output in all cases] {param: NIL}
[Explanation] These refer to specific subtypes of clearance (e.g., dialysis, uptake, biliary) and not to CL parameters in our pk_ontology.

[Mention] renal excretion rate (1.69 microg x min(-1))
[Expected Output] {param: NIL}
[Explanation] We are not interested in rates, only rate constants, link to NIL. Remember, the units can give clues e.g. this is not kexcr which has units of 1/time. 

[Mention] unbound renal clearance
[Expected Output] {param: CLu}
[Explanation] Refers to clearance of the unbound fraction of drug, links to CLu.

[Mention] AUC12 hours
[Expected Output] {param: AUCt}
[Explanation] The mention refers to area under the curve over a fixed time window (0-12h), matching AUCt.
"""


TABLE_EXAMPLES = """
The following examples are provided to guide you. They include mentions (with context if relevant), expected answers and explanations: 

[Mention] Vd,ss/F (mL/kg) 
[Expected Output] {param: NIL}
[Explanation] This does not fit exactly with any a parameter in the pk_ontology (closest to V/F but at steady-state) -> return NIL.  

[Mention] Cmax at steady-state
[Expected Output] {param: NIL}
[Explanation] This does not fit exactly with any a parameter in the pk_ontology (closest to Cmax but at steady-state) -> return NIL.    

[Mention] ARCtrough 
[FOOTER] ARCtrough = Accumulation ratio of trough concentrations
[Expected Output] {param: NIL}
[Explanation] Does not link specifically to any entries in the pk_ontology -> return NIL.

[Mention] Cavg0–336h
[Expected Output] {param: Cavg}
[Explanation] Average concentration even over a defined interval, should link to Cavg.

[Mention] CL0–t  hour,secretion (mL/min)d
[FOOTER] CL, clearance; Cmax, maximum plasma concentration.
[Expected Output] {param: CL}
[Explanation]  Clearance, even over a defined interval, should link to CL.   

[Mentions] AUC0–∞/D, AUC0–∞, norm 
[Expected Output in both cases] {param: AUC/dose}
[Explanation] Even though "AUC0–∞" would normally link to AUC∞, both mentions are dose-normalized, so must link to AUC/dose.

[Mention] CLM,NR (L/hr)
[FOOTER] CL, clearance; M3G, morphine‐3‐glucuronide.
[Expected Output] {param: CLNR}
[Explanation] "NR" represent non-renal in this case and so is critical for linking.    

[Mention] VM
[FOOTER] Definitions: VM = Central volume of distribution of morphine (M).
[Expected Output] {param: V1}
[Explanation] The footer shows that M refers to the drug name and should be ignored for linking. 
"""

class ZeroShotPromptEntityLinker:

  def __init__(self, generative_model):
      self.generative_model = generative_model

  def prompt(self, user_text, system_text, model: str = None):
        if model is None:
            raise ValueError("Please specify a model")

        if "gpt" in model.lower():
            client = openai
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': system_text},
                    {'role': 'user', 'content': user_text}
                ],
                temperature=0
            )
            answer = response.choices[0].message.content
        else:
            raise ValueError(f"Model {model} is not supported.")

        return answer

  def ground(self, mention: str, kb_concepts: str, system_prompt: str, param_to_id, model: str, context=None, examples=None):
      pr = self.gen_prompt(
          mention=mention,
          kb_concepts=kb_concepts,
          system_prompt=system_prompt,
          context=context,
          examples=examples,
      )
      a = 1
      response = self.prompt(user_text=pr["user_prompt"], system_text=pr["system_prompt"], model=model)
      try:
          answer = response.replace("param:", "").strip("{}").strip().lower()
          result = {}
          param_to_id_lower = {k.lower().strip(): v for k, v in param_to_id.items()}

          if answer in ["nil", "n/a", "none"]:
              result["id"] = "Q100"
          elif answer in param_to_id_lower:
              result["id"] = param_to_id_lower[answer]
          else:
              print(f"\n Unrecognized LLM answer: '{answer}' for mention: '{mention}', assigning NIL.")
              #print(f"🔍 Raw response: {response}")
              #print(f"🔎 Prompt preview:\n---\n{pr['system_prompt']}\n{pr['user_prompt'][:300]}...\n---")
              return None

      except Exception as e:
          print(f"❌ Error processing response: {e}")
          #result = {"id": "Q100"}  # also fallback in case of hard failure
          return None

      return result

  def gen_prompt(self, mention: str, kb_concepts: str, system_prompt: str, context=None, examples=None):
      """
          Prepares the structured prompt for entity linking.

          Args:
              mention (str): Mention text (already optionally marked with <mention>...</mention>)
              kb_concepts (str): Pre-formatted knowledge base string
              system_prompt (str): The base instruction prompt
              context (list, optional): List of context strings (e.g., row, col, footer info)
              examples (list, optional): List of example formatted strings

          Returns:
              dict: {'system_prompt': system_prompt, 'user_prompt': user_prompt}
      """
      user_prompt = ""

      user_prompt += f"\n---\nMention:\n{mention}\n---\n"

      if context:
          user_prompt += f"\n---\nContext:\n{context}\n---\n"

      if examples:
          user_prompt += f"\n---\nTask Examples:\n{examples}\n---\n"

      user_prompt += f"\n---\nOntology:\n\n{kb_concepts}\n---\n"

      return {
          "system_prompt": system_prompt.strip(),
          "user_prompt": user_prompt.strip()
      }


def merge_llm_and_unmatched_results(llm_results, unmatched_examples):
    for ex in unmatched_examples:
        llm_results["y_true"].append(ex["label"])
        llm_results["y_pred"].append(ex["llm_pred"])
        llm_results["texts"].append(ex.get("text_with_tagged_mention", ex["mention"]))
    return llm_results


def link_mentions_with_llm(
    dataset: List[dict],
    model_name: str,
    system_prompt: str,
    param_to_id: dict,
    text_key: str = "mention",
    examples: str =None,
    context_key: str =None,
):
    """
    Link mentions from a dataset to pk_ontology parameters using a zero-shot LLM linker.

    Args:
        dataset (List[dict]): List of mention dictionaries (must have 'mention', 'text_with_tagged_mention', etc.)
        model_name (str): Model to use (e.g., "gpt-4").
        system_prompt (str): System instruction prompt.
        param_to_id (dict): Mapping from normalized parameter names → IDs.

    Returns:
        dict: Evaluation and predictions (y_true, y_pred, texts, no_answer)
    """
    y_true = []
    y_pred = []
    texts = []
    no_answer = []

    # Init linker
    zsel = ZeroShotPromptEntityLinker(generative_model=model_name)

    print("🔎 Processing dataset...")

    #linking_examples = None
    for example in tqdm(dataset):
        mention = example[text_key]
        labels = example["label"]
        kb_concepts = example["ontology_subset"]

        if context_key:
            context = example[context_key]
        else:
            context = None

        # Link!
        answer = zsel.ground(
            mention=mention,
            kb_concepts=kb_concepts,
            model=model_name,
            system_prompt=system_prompt,
            context=context,
            examples=examples,
            param_to_id=param_to_id,
        )

        if answer is not None and "id" in answer:
            if context_key and context_key != "text_with_tagged_mention":
                text_for_error = example["text_with_tagged_mention"] + example.get(context_key, "")
            else:
                text_for_error = example["text_with_tagged_mention"]

            y_pred.append(answer["id"])
            texts.append(text_for_error)
            y_true.append(labels)

        else:
            # Grounding failed completely
            y_pred.append("Q100")
            y_true.append(labels)
            texts.append(example.get("text_with_tagged_mention", mention))
            no_answer.append(mention)

    print(f"❗ Total with no valid answer: {len(no_answer)}")

    return {
        "model_name": model_name,
        "y_true": y_true,
        "y_pred": y_pred,
        "texts": texts,
        "no_answer": no_answer
    }


def evaluate_llm_runs(dataset, model_name, text_key, context_key, system_prompt, param_to_id, examples, id_to_label, n_runs=3, split_results=True, plot_conf_matrix=True):
    all_f1 = []
    match_counts = []
    all_results = []
    all_dev_results = []
    all_unlinked_results = []
    all_y_true = []
    all_y_pred = []

    for i in range(n_runs):
        print(f"\n🔁 LLM Run {i + 1}/{n_runs}")
        results = link_mentions_with_llm(
            dataset=dataset,
            model_name=model_name,
            context_key=context_key,
            system_prompt=system_prompt,
            text_key=text_key,
            param_to_id=param_to_id,
            examples=examples,
        )
        all_results.append(results)

        y_true = results["y_true"]
        y_pred = results["y_pred"]
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        eval_result = evaluate(y_true, y_pred, id_to_label=id_to_label)
        mic_f1 = float(eval_result["mic_F1"].replace("%", ""))
        all_f1.append(mic_f1)

        num_matched = len(dataset) - len(results["no_answer"])
        match_counts.append(num_matched)

        if split_results:
            results_breakdown = split_and_eval_by_source(results, dataset, param_to_id)
            all_dev_results.append(results_breakdown["dev"]["mic_f1"])
            all_unlinked_results.append(results_breakdown["unlinked"]["mic_f1"])

    # ✅ Plot confusion matrix after all runs
    if plot_conf_matrix:
        plot_confusion_matrix(all_y_true, all_y_pred, label_mapping=id_to_label)

    # ✅ Summarise
    mean_f1 = np.mean(all_f1)
    std_f1 = np.std(all_f1)
    mean_matched = int(np.mean(match_counts))
    print(f"\n✅ Avg. matched examples across {n_runs} runs: {mean_matched}/{len(dataset)}")

    if split_results:
        return {
            "all_f1": all_f1,
            "mean_f1": mean_f1,
            "std_f1": std_f1,
            "mean_matched": mean_matched,
            "runs": all_results,
            "mean_dev_f1": np.mean(all_dev_results),
            "std_dev_f1": np.std(all_dev_results),
            "mean_unlinked_mic_f1": np.mean(all_unlinked_results),
            "std_unlinked_mic_f1": np.std(all_unlinked_results),
        }
    else:
        return {
            "all_f1": all_f1,
            "mean_f1": mean_f1,
            "std_f1": std_f1,
            "mean_matched": mean_matched,
            "runs": all_results,
        }



def remove_unlinked(full_data, unlinked_data):
    def unique_key(ex):
        return (ex.get("text"), ex["spans"][0]["start"])  # or another reliable combo

    unlinked_keys = {unique_key(ex) for ex in unlinked_data}
    return [ex for ex in full_data if unique_key(ex) not in unlinked_keys]


def split_and_eval_by_source(results, combined_dataset, id_to_label):
    results_by_source = defaultdict(lambda: {"y_true": [], "y_pred": []})

    for ex, y_true, y_pred in zip(combined_dataset, results["y_true"], results["y_pred"]):
        source = ex.get("source", "unknown")
        results_by_source[source]["y_true"].append(y_true)
        results_by_source[source]["y_pred"].append(y_pred)

    eval_summaries = {}
    for source, group in results_by_source.items():
        print(f"\n\nSource: {source}")
        metrics = evaluate(group["y_true"], group["y_pred"], id_to_label=id_to_label)
        eval_summaries[source] = {
            "mic_f1": float(metrics["mic_F1"].replace("%", "")),
            "n": len(group["y_true"])
        }
        print("\n")

    return eval_summaries



def estimate_average_tokens_and_cost(
    dataset: List[dict],
    model: str,
    system_prompt: str,
    examples: str = None,
    context_key: str = None,
    est_completion_tokens=7,
    print_summary=True,
):

    # Tokenizer
    encoding = tiktoken.encoding_for_model(model)

    # Pricing (per 1M tokens)
    prices = {
        "gpt-4o": (0.0025, 0.01),
        "gpt-4o-mini": (0.00015, 0.0006),
        "gpt-4.1": (0.002, 0.008),
        "gpt-4.1-mini": (0.0004, 0.0016),
        "gpt-4.1-nano": (0.0001, 0.0004),
    }

    if model not in prices:
        raise ValueError(f"Unsupported model: {model}")

    prompt_rate, completion_rate = prices[model]

    total_prompt_tokens = 0
    total_cost = 0

    zsel = ZeroShotPromptEntityLinker(generative_model=model)

    for ex in dataset:
        mention = ex["mention"]
        kb_concepts = ex["ontology_subset"]
        context = ex.get(context_key) if context_key else None

        prompt_parts = zsel.gen_prompt(
            mention=mention,
            kb_concepts=kb_concepts,
            system_prompt=system_prompt,
            context=context,
            examples=examples,
        )

        full_prompt = prompt_parts["system_prompt"] + "\n" + prompt_parts["user_prompt"]
        prompt_tokens = len(encoding.encode(full_prompt))
        total_prompt_tokens += prompt_tokens

        # Estimate cost for this example
        cost = (prompt_tokens / 1000 * prompt_rate) + (est_completion_tokens / 1000 * completion_rate)
        total_cost += cost

    n = len(dataset)
    avg_tokens = round(total_prompt_tokens / n, 2)
    avg_cost = round(total_cost / n, 6)
    total_cost = round(total_cost, 6)
    cost_per_1000 = round(avg_cost * 1000, 6)

    result = {
        "model": model,
        "total_examples": n,
        "average_prompt_tokens": avg_tokens,
        "average_cost_per_example": avg_cost,
        "cost_per_1000_examples": cost_per_1000,
        "estimated_total_cost": total_cost,
        "estimated_completion_tokens": est_completion_tokens,
    }

    if print_summary:
        print("\n📊 Estimated Token & Cost Summary")
        print("────────────────────────────────────────")
        print(f"Model: {result['model']}")
        print(f"Examples evaluated: {result['total_examples']}")
        print(f"Avg. prompt tokens per example: {result['average_prompt_tokens']}")
        print(f"Cost per 1000 examples: {result['cost_per_1000_examples']:.2f}")
        print(f"Estimated completion tokens: {result['estimated_completion_tokens']}")
        print(f"Avg. cost per example: ${result['average_cost_per_example']}")
        print(f"Total estimated cost: ${result['estimated_total_cost']}")
        print("────────────────────────────────────────\n")

    return result


"""
    Testing the idea of refinfining membership by constructing members based on increasing "distance" from true members.
    Try both actual edit distance, and neighbors generated by the MIA attack (more semantically similar)
"""
import numpy as np
import torch
from tqdm import tqdm
import random
import datetime
import os
import json
import pickle
import math
from collections import defaultdict
from functools import partial
from typing import List

from simple_parsing import ArgumentParser
from pathlib import Path

from mimir.config import (
    ExperimentConfig,
    EnvironmentConfig,
    NeighborhoodConfig,
    ReferenceConfig,
    OpenAIConfig,
)
import mimir.data_utils as data_utils
import mimir.plot_utils as plot_utils
from mimir.models import LanguageModel, ReferenceModel, OpenAI_APIModel
from mimir.attacks.blackbox_attacks import BlackBoxAttacks
from mimir.attacks.neighborhood import T5Model, BertModel, NeighborhoodAttack
from mimir.attacks.attack_utils import (
    f1_score,
    get_roc_metrics,
    get_precision_recall_metrics,
    get_auc_from_thresholds,
)

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def run_blackbox_attacks(
    data,
    ds_objects,
    target_model,
    ref_models,
    config: ExperimentConfig,
    n_samples: int = None,
    batch_size: int = 50,
    keys_care_about: List[str] = ["nonmember", "member"],
    scores_not_needed: bool = False,
    verbose: bool = True,
):
    torch.manual_seed(0)
    np.random.seed(0)

    n_samples = len(data["nonmember"]) if n_samples is None else n_samples

    # Structure: attack -> member scores/nonmember scores
    # For both members and nonmembers, we compute all attacks
    # listed in config all together for each sample
    attacks = config.blackbox_attacks
    neigh_config = config.neighborhood_config
    implemented_blackbox_attacks = [a.value for a in BlackBoxAttacks]
    # check for unimplemented attacks
    runnable_attacks = []
    for a in attacks:
        if a not in implemented_blackbox_attacks:
            print(f"Attack {a} not implemented, will be ignored")
            pass

        runnable_attacks.append(a)
    attacks = runnable_attacks

    if BlackBoxAttacks.NEIGHBOR in attacks:
        neighborhood_attacker = NeighborhoodAttack(config, target_model)
        neighborhood_attacker.prepare()

    results = defaultdict(list)
    for classification in keys_care_about:
        print(f"Running for classification {classification}")

        neighbors = None
        if BlackBoxAttacks.NEIGHBOR in attacks and neigh_config.load_from_cache:
            neighbors = data[f"{classification}_neighbors"]
            print("Loaded neighbors from cache!")

        collected_neighbors = {
            n_perturbation: [] for n_perturbation in n_perturbation_list
        }

        # For each batch of data
        # TODO: Batch-size isn't really "batching" data - change later
        iterator = range(math.ceil(n_samples / batch_size))
        if verbose:
            iterator = tqdm(iterator, desc=f"Computing criterion")

        for batch in iterator:
            texts = data[classification][batch * batch_size : (batch + 1) * batch_size]

            # For each entry in batch
            for idx in range(len(texts)):
                sample_information = defaultdict(list)
                sample = (
                    texts[idx][: config.max_substrs]
                    if config.full_doc
                    else [texts[idx]]
                )

                # This will be a list of integers if pretokenized
                sample_information["sample"] = sample
                if config.pretokenized:
                    detokenized_sample = [
                        target_model.tokenizer.decode(s) for s in sample
                    ]
                    sample_information["detokenized"] = detokenized_sample

                # For each substring
                neighbors_within = {
                    n_perturbation: [] for n_perturbation in n_perturbation_list
                }
                for i, substr in enumerate(sample):
                    # compute token probabilities for sample
                    s_tk_probs = (
                        target_model.get_probabilities(substr)
                        if not config.pretokenized
                        else target_model.get_probabilities(
                            detokenized_sample[i], tokens=substr
                        )
                    )

                    # always compute loss score
                    loss = (
                        target_model.get_ll(substr, probs=s_tk_probs)
                        if not config.pretokenized
                        else target_model.get_ll(
                            detokenized_sample[i], tokens=substr, probs=s_tk_probs
                        )
                    )
                    # TODO: Instead of doing this outside, set config default to always include LOSS
                    sample_information[BlackBoxAttacks.LOSS].append(loss)

                    # TODO: Shift functionality into each attack entirely, so that this is just a for loop
                    # For each attack
                    for attack in attacks:
                        if attack == BlackBoxAttacks.ZLIB:
                            score = (
                                target_model.get_zlib_entropy(substr, probs=s_tk_probs)
                                if not config.pretokenized
                                else target_model.get_zlib_entropy(
                                    detokenized_sample[i],
                                    tokens=substr,
                                    probs=s_tk_probs,
                                )
                            )
                            sample_information[attack].append(score)
                        elif attack == BlackBoxAttacks.MIN_K:
                            score = (
                                target_model.get_min_k_prob(substr, probs=s_tk_probs)
                                if not config.pretokenized
                                else target_model.get_min_k_prob(
                                    detokenized_sample[i],
                                    tokens=substr,
                                    probs=s_tk_probs,
                                )
                            )
                            sample_information[attack].append(score)
                        elif attack == BlackBoxAttacks.NEIGHBOR:
                            # For each 'number of neighbors'
                            for n_perturbation in n_perturbation_list:
                                # Use neighbors if available
                                if neighbors:
                                    substr_neighbors = neighbors[n_perturbation][
                                        batch * batch_size + idx
                                    ][i]
                                else:
                                    substr_neighbors = (
                                        neighborhood_attacker.get_neighbors(
                                            [substr], n_perturbations=n_perturbation
                                        )
                                    )
                                    # Collect this neighbor information if neigh_config.dump_cache is True
                                    if neigh_config.dump_cache:
                                        neighbors_within[n_perturbation].append(
                                            substr_neighbors
                                        )

                                if not neigh_config.dump_cache:
                                    # Only evaluate neighborhood attack when not caching neighbors
                                    mean_substr_score = target_model.get_lls(
                                        substr_neighbors, batch_size=15
                                    )
                                    d_based_score = loss - mean_substr_score

                                    sample_information[
                                        f"{attack}-{n_perturbation}"
                                    ].append(d_based_score)

                if neigh_config and neigh_config.dump_cache:
                    for n_perturbation in n_perturbation_list:
                        collected_neighbors[n_perturbation].append(
                            neighbors_within[n_perturbation]
                        )

                # Add the scores we collected for each sample for each
                # attack into to respective list for its classification
                results[classification].append(sample_information)

        if neigh_config and neigh_config.dump_cache:
            ds_obj_use = ds_objects[classification]

            # Save p_member_text and p_nonmember_text (Lists of strings) to cache
            # For each perturbation
            for n_perturbation in n_perturbation_list:
                ds_obj_use.dump_neighbors(
                    collected_neighbors[n_perturbation],
                    train=True if classification == "member" else False,
                    num_neighbors=n_perturbation,
                    model=neigh_config.model,
                    in_place_swap=in_place_swap,
                )

    if neigh_config and neigh_config.dump_cache:
        print(
            "Data dumped! Please re-run with load_from_cache set to True in neigh_config"
        )
        exit(0)

    # Perform reference-based attacks
    if BlackBoxAttacks.REFERENCE_BASED in attacks:
        if ref_models is None:
            print("No reference models specified, skipping Reference-based attacks")
        else:
            for name, ref_model in ref_models.items():
                # if "llama" not in name and "alpaca" not in name:
                ref_model.load()

                # Update collected scores for each sample with ref-based attack scores
                for classification, result in results.items():
                    itr = result
                    if verbose:
                        itr = tqdm(itr, desc="Ref scores")
                    for r in itr:
                        ref_model_scores = []
                        for i, s in enumerate(r["sample"]):
                            if config.pretokenized:
                                s = r["detokenized"][i]
                            ref_score = r[BlackBoxAttacks.LOSS][i] - ref_model.get_ll(s)
                            ref_model_scores.append(ref_score)
                        r[
                            f"{BlackBoxAttacks.REFERENCE_BASED}-{name.split('/')[-1]}"
                        ].extend(ref_model_scores)

                # if "llama" not in name and "alpaca" not in name:
                ref_model.unload()

    # Rearrange the nesting of the results dict and calculated aggregated score for sample
    # attack -> member/nonmember -> list of scores
    samples = defaultdict(list)
    predictions = defaultdict(lambda: defaultdict(list))
    for classification, result in results.items():
        for r in result:
            samples[classification].append(r["sample"])
            for attack, scores in r.items():
                if attack != "sample" and attack != "detokenized":
                    predictions[attack][classification].append(np.min(scores))

    if scores_not_needed:
        return predictions

    # Collect outputs for each attack
    blackbox_attack_outputs = {}
    for attack, prediction in tqdm(predictions.items()):
        fpr, tpr, roc_auc, roc_auc_res, thresholds = get_roc_metrics(
            preds_member=prediction["member"],
            preds_nonmember=prediction["nonmember"],
            perform_bootstrap=True,
            return_thresholds=True,
        )
        tpr_at_low_fpr = {
            upper_bound: tpr[np.where(np.array(fpr) < upper_bound)[0][-1]]
            for upper_bound in config.fpr_list
        }
        p, r, pr_auc = get_precision_recall_metrics(
            preds_member=prediction["member"], preds_nonmember=prediction["nonmember"]
        )

        print(
            f"{attack}_threshold ROC AUC: {roc_auc}, PR AUC: {pr_auc}, tpr_at_low_fpr: {tpr_at_low_fpr}"
        )
        blackbox_attack_outputs[attack] = {
            "name": f"{attack}_threshold",
            "predictions": prediction,
            "info": {
                "n_samples": n_samples,
            },
            "raw_results": samples if not config.pretokenized else [],
            "metrics": {
                "roc_auc": roc_auc,
                "fpr": fpr,
                "tpr": tpr,
                "bootstrap_roc_auc_mean": np.mean(roc_auc_res.bootstrap_distribution),
                "bootstrap_roc_auc_std": roc_auc_res.standard_error,
                "tpr_at_low_fpr": tpr_at_low_fpr,
                "thresholds": thresholds,
            },
            "pr_metrics": {
                "pr_auc": pr_auc,
                "precision": p,
                "recall": r,
            },
            "loss": 1 - pr_auc,
        }

    return blackbox_attack_outputs


def generate_data_processed(
    raw_data_member, batch_size, raw_data_non_member: List[str] = None
):
    torch.manual_seed(42)
    np.random.seed(42)
    data = {
        "nonmember": [],
        "member": [],
    }

    seq_lens = []
    num_batches = (len(raw_data_member) // batch_size) + 1
    iterator = tqdm(range(num_batches), desc="Generating samples")
    for batch in iterator:
        member_text = raw_data_member[batch * batch_size : (batch + 1) * batch_size]
        non_member_text = raw_data_non_member[
            batch * batch_size : (batch + 1) * batch_size
        ]

        # TODO make same len
        for o, s in zip(non_member_text, member_text):
            # o, s = data_utils.trim_to_shorter_length(o, s, config.max_words)

            # # add to the data
            # assert len(o.split(' ')) == len(s.split(' '))
            if not config.full_doc:
                seq_lens.append((len(s.split(" ")), len(o.split())))

            if config.tok_by_tok:
                for tok_cnt in range(len(o.split(" "))):
                    data["nonmember"].append(" ".join(o.split(" ")[: tok_cnt + 1]))
                    data["member"].append(" ".join(s.split(" ")[: tok_cnt + 1]))
            else:
                data["nonmember"].append(o)
                data["member"].append(s)

    # if config.tok_by_tok:
    n_samples = len(data["nonmember"])
    # else:
    #     n_samples = config.n_samples
    if config.pre_perturb_pct > 0:
        print(
            f"APPLYING {config.pre_perturb_pct}, {config.pre_perturb_span_length} PRE-PERTURBATIONS"
        )
        print("MOVING MASK MODEL TO GPU...", end="", flush=True)
        mask_model.load()
        data["member"] = mask_model.generate_neighbors(
            data["member"],
            config.pre_perturb_span_length,
            config.pre_perturb_pct,
            config.chunk_size,
            ceil_pct=True,
        )
        print("MOVING BASE MODEL TO GPU...", end="", flush=True)
        base_model.load()

    return data, seq_lens, n_samples


def generate_data(
    dataset: str,
    train: bool = True,
    presampled: str = None,
    specific_source: str = None,
):
    data_obj = data_utils.Data(dataset, config=config, presampled=presampled)
    data = data_obj.load(
        train=train,
        mask_tokenizer=mask_model.tokenizer if mask_model else None,
        specific_source=specific_source,
    )
    return data_obj, data
    # return generate_samples(data[:n_samples], batch_size=batch_size)


if __name__ == "__main__":
    # TODO: Shift below to main() function - variables here are global and may interfe with functions etc.

    # Extract relevant configurations from config file
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--config", help="Path to attack config file", type=Path)
    args, remaining_argv = parser.parse_known_args()
    # Attempt to extract as much information from config file as you can
    config = ExperimentConfig.load(args.config, drop_extra_fields=False)
    # Also give user the option to provide config values over CLI
    parser = ArgumentParser(parents=[parser])
    parser.add_arguments(ExperimentConfig, dest="exp_config", default=config)
    args = parser.parse_args(remaining_argv)
    config: ExperimentConfig = args.exp_config

    env_config: EnvironmentConfig = config.env_config
    neigh_config: NeighborhoodConfig = config.neighborhood_config
    ref_config: ReferenceConfig = config.ref_config
    openai_config: OpenAIConfig = config.openai_config

    if neigh_config:
        if neigh_config.load_from_cache and neigh_config.dump_cache:
            raise ValueError(
                "Cannot dump and load from cache at the same time. Please set one of these to False"
            )

    if openai_config:
        openAI_model = OpenAI_APIModel(config)

    if openai_config is not None:
        import openai

        assert openai_config.key is not None, "Must provide OpenAI API key"
        openai.api_key = openai_config.key

    START_DATE = datetime.datetime.now().strftime("%Y-%m-%d")
    START_TIME = datetime.datetime.now().strftime("%H-%M-%S-%f")

    # define SAVE_FOLDER as the timestamp - base model name - mask filling model name
    # create it if it doesn't exist
    precision_string = (
        "int8" if env_config.int8 else ("fp16" if env_config.half else "fp32")
    )
    sampling_string = (
        "top_k" if config.do_top_k else ("top_p" if config.do_top_p else "temp")
    )
    output_subfolder = f"{config.output_name}/" if config.output_name else ""
    if openai_config is None:
        base_model_name = config.base_model.replace("/", "_")
    else:
        base_model_name = "openai-" + openai_config.model.replace("/", "_")
    scoring_model_string = (
        f"-{config.scoring_model_name}" if config.scoring_model_name else ""
    ).replace("/", "_")

    if config.tok_by_tok:
        tok_by_tok_string = "--tok_true"
    else:
        tok_by_tok_string = "--tok_false"

    if neigh_config:
        if neigh_config.span_length == 2:
            span_length_string = ""
        else:
            span_length_string = f"--{neigh_config.span_length}"

    # Replace paths
    dataset_member_name = config.dataset_member.replace("/", "_")
    dataset_nonmember_name = config.dataset_nonmember.replace("/", "_")

    sf_ext = "new_mi_"

    default_prompt_len = 30
    # suffix = "QUANTILE_TEST"
    # TODO - Either automate suffix construction, or use better names (e.g. save folder with results, and attack config in it)
    suffix = f"{sf_ext}{output_subfolder}{base_model_name}-{scoring_model_string}-{neigh_config.model}-{sampling_string}/{precision_string}-{neigh_config.pct_words_masked}-{neigh_config.n_perturbation_rounds}-{dataset_member_name}-{dataset_nonmember_name}-{config.n_samples}{span_length_string}{config.max_words}{config.min_words}_plen{default_prompt_len}_{tok_by_tok_string}"
    suffix += "--shifted_mi_defn_exps"

    # Add pile source to suffix, if provided
    # TODO: Shift dataset-specific processing to their corresponding classes
    if config.specific_source is not None:
        processed_source = data_utils.sourcename_process(config.specific_source)
        suffix += f"-{processed_source}"
    SAVE_FOLDER = os.path.join(env_config.tmp_results, suffix)

    new_folder = os.path.join(env_config.results, suffix)
    ##don't run if exists!!!
    print(f"{new_folder}")
    if os.path.isdir((new_folder)):
        print(f"HERE folder exists, not running this exp {new_folder}")
        exit(0)

    if not (os.path.exists(SAVE_FOLDER) or config.dump_cache):
        os.makedirs(SAVE_FOLDER)
    print(f"Saving results to absolute path: {os.path.abspath(SAVE_FOLDER)}")

    # write args to file
    # if not config.dump_cache:
    #     config.save(os.path.join(SAVE_FOLDER, 'args.json'), indent=4)

    if neigh_config:
        n_perturbation_list = neigh_config.n_perturbation_list
        n_perturbation_rounds = neigh_config.n_perturbation_rounds
        in_place_swap = neigh_config.original_tokenization_swap
        # n_similarity_samples = args.n_similarity_samples # NOT USED

    cache_dir = env_config.cache_dir
    os.environ["XDG_CACHE_HOME"] = cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print(f"Using cache dir {cache_dir}")

    # generic generative model
    base_model = LanguageModel(config)

    # reference model if we are doing the ref-based attack
    ref_models = None
    if (
        ref_config is not None
        and BlackBoxAttacks.REFERENCE_BASED in config.blackbox_attacks
    ):
        ref_models = {
            model: ReferenceModel(config, model) for model in ref_config.models
        }
        # print('MOVING ref MODEL TO GPU...', end='', flush=True)

    # Load neighborhood attack model, only if we are doing the neighborhood attack AND generating neighbors
    mask_model = None
    if (
        neigh_config
        and (not neigh_config.load_from_cache)
        and (BlackBoxAttacks.NEIGHBOR in config.blackbox_attacks)
    ):
        model_kwargs = dict()
        if not config.baselines_only and not neigh_config.random_fills:
            if env_config.int8:
                model_kwargs = dict(
                    load_in_8bit=True, device_map="auto", torch_dtype=torch.bfloat16
                )
            elif env_config.half:
                model_kwargs = dict(torch_dtype=torch.bfloat16)
            try:
                n_positions = 512  # Should fix later, but for T-5 this is 512 indeed
                # mask_model.config.n_positions
            except AttributeError:
                n_positions = config.max_tokens
        else:
            n_positions = config.max_tokens
        tokenizer_kwargs = {
            "model_max_length": n_positions,
        }
        print(f"Loading mask filling model {config.neighborhood_config.model}...")
        if "t5" in config.neighborhood_config.model:
            mask_model = T5Model(
                config, model_kwargs=model_kwargs, tokenizer_kwargs=tokenizer_kwargs
            )
        elif "bert" in config.neighborhood_config.model:
            mask_model = BertModel(config)
        else:
            raise ValueError(f"Unknown model {config.neighborhood_config.model}")
        # if config.dataset_member in ['english', 'german']:
        #     preproc_tokenizer = mask_tokenizer

    print("MOVING BASE MODEL TO GPU...", end="", flush=True)
    base_model.load()

    print(f"Loading dataset {config.dataset_member} and {config.dataset_nonmember}...")
    # data, seq_lens, n_samples = generate_data(config.dataset_member)

    data_obj_mem, data_member = generate_data(
        config.dataset_member, presampled=config.presampled_dataset_member
    )

    ### <LOGIC FOR SPECIFIC EXPERIMENTS>
    def edit(x, n: int):
        """
        Return version of x that has some distance 'n' from x.
        Could be edit-distance based, or semantic distance (NE) based
        """
        # Tokenize sentence
        x_tok = base_model.tokenizer(x)["input_ids"]
        # Pick n random positions
        positions = np.random.choice(len(x_tok), n, replace=False)
        # Replace those positions with random words from vocabulary
        for pos in positions:
            x_tok[pos] = np.random.choice(base_model.tokenizer.vocab_size)
        # Detokenize
        x = base_model.tokenizer.decode(x_tok)
        return x

    # """
    with open(
        f"/p/distinf/uw_llm_collab/edit_distance_members/{config.specific_source}.json",
        "r",
    ) as f:
        other_members_data = json.load(f)
        n_try = list(other_members_data.keys())
        n_trials = len(other_members_data[n_try[0]])
    # """

    """
    # Try out multiple "distances"
    n_try = [1, 5, 10, 25, 100]
    # With multiple trials
    n_trials = 20
    other_members_data = {}
    for n in tqdm(n_try, "Generating edited members"):
        trials = {}
        for i in tqdm(range(n_trials)):
            trials[i] = [edit(x, n) for x in data_member]
        other_members_data[n] = trials
    with open(
        f"/p/distinf/uw_llm_collab/edit_distance_members/{config.specific_source}.json",
        "w",
    ) as f:
        json.dump(other_members_data, f)
    print("Data dumped! Please re-run with load_from_cache set to True")
    exit(0)
    """
    ### </LOGIC FOR SPECIFIC EXPERIMENTS>

    # Using thresholds returned in blackbox_outputs, compute AUCs and ROC curves for other non-member sources
    score_dict = {x: [] for x in config.blackbox_attacks}
    for k, v in score_dict.items():
        score_dict[k] = {x: {} for x in n_try}

    pbar = tqdm(total=len(n_try) * n_trials)
    for n, other_member in other_members_data.items():
        ds_objects = {"member": data_obj_mem}
        for i in range(n_trials):
            n_samples = len(other_member[str(i)])
            other_blackbox_predictions = run_blackbox_attacks(
                data={"member": other_member[str(i)]},
                ds_objects=ds_objects,
                target_model=base_model,
                ref_models=ref_models,
                config=config,
                n_samples=n_samples,
                keys_care_about=["member"],
                scores_not_needed=True,
                verbose=False,
            )
            pbar.update(1)

            for attack in config.blackbox_attacks:
                score_dict[attack][n][i] = other_blackbox_predictions[attack]["member"]

    pbar.close()
    with open(f"./edit_distance_results_{args.specific_source}.json", "w") as f:
        json.dump(score_dict, f)

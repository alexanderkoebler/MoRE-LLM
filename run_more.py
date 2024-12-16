"""
Copyright 2024 Siemens AG

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import json
import time
import datetime
import argparse
import numpy as np

from tqdm import trange

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from alibi.explainers import AnchorTabular

from torch.utils.tensorboard import SummaryWriter

from more import *
from models import *
from utils import set_seed


def log_performance(moe, X_test, y_test, L_star, L_train, L_test, name, preprocessor, r_file=None, th=0.5):
    loss_task_func = nn.CrossEntropyLoss()

    output = moe.rule_expert(torch.tensor(preprocessor.inverse_transform(X_test)))
    loss_rules = loss_task_func(output.squeeze(), y_test.long())

    mask = torch.sum(output,axis=-1).bool().numpy()
    y_gate = moe.gating_model(X_test).detach().numpy()
    gate_coverage = (np.sum(y_gate[:,1] > 0.5)/len(y_gate))*100
    gate_coverage_in_rules = (np.sum(y_gate[mask,1] > 0.5)/len(y_gate))*100

    rule_coverage = calculate_coverage_of_ruleset(output)*100
    rule_accuracy = calculate_accuracy_of_ruleset(output, y_test)*100
    num_rules = len(moe.rule_expert.rules)

    acc = accuracy_score(y_test.detach().cpu().numpy(), moe(X_test)[0].argmax(dim=1).detach().cpu().numpy())
    roc = roc_auc_score(y_test.detach().cpu().numpy(), moe(X_test)[0].argmax(dim=1).detach().cpu().numpy())

    acc_hard = accuracy_score(y_test.detach().cpu().numpy(), moe.predict_hard(X_test,th)[0].argmax(dim=1).detach().cpu().numpy())
    roc_hard = roc_auc_score(y_test.detach().cpu().numpy(), moe.predict_hard(X_test,th)[0].argmax(dim=1).detach().cpu().numpy())

    if np.sum(mask):
        acc_rule_instances = sklearn.metrics.accuracy_score(y_test[mask], np.argmax(moe(X_test[mask])[0].detach().numpy(),axis=1))
    else:
        acc_rule_instances = 0.0

    if r_file != None:
        r_file.write("# {}".format(name) + "\n")
        r_file.write("{} loss star {}".format(name, L_star) + "\n")
        r_file.write("{} loss train {}".format(name, L_train) + "\n")
        r_file.write("{} loss test {}".format(name, L_test) + "\n")
        r_file.write("{} accuracy {}".format(name, acc) + "\n")
        r_file.write("{} rocauc {}".format(name, roc) + "\n")
        r_file.write("{} accuracy hard {} (th={})".format(name, acc_hard, th) + "\n")
        r_file.write("{} rocauc hard {} (th={})".format(name, roc_hard, th) + "\n")
        r_file.write('{} coverage of rules: {:.2f}%'.format(name, rule_coverage) + "\n")
        r_file.write('{} accuracy of rules: {:.2f}%'.format(name, rule_accuracy) + "\n")
        r_file.write('{} loss of rules: {:.2f}'.format(name, loss_rules) + "\n")
        r_file.write('{} coverage of gate in rules: {:.2f}%'.format(name, gate_coverage_in_rules) + "\n")
        r_file.write('{} coverage of gate: {:.2f}%'.format(name, gate_coverage) + "\n")
        r_file.write('{} accuracy on instances covered by rule set {}'.format(name, acc_rule_instances) + "\n")
        r_file.write('{} num rules {}'.format(name, num_rules) + "\n")

    print("# {}".format(name) + "\n")
    print("{} loss star {}".format(name, L_star) + "\n")
    print("{} loss train {}".format(name, L_train) + "\n")
    print("{} loss test {}".format(name, L_test) + "\n")
    print("{} accuracy {}".format(name, acc) + "\n")
    print("{} rocauc {}".format(name, roc) + "\n")
    print("{} accuracy hard {} (th={})".format(name, acc_hard, th) + "\n")
    print("{} rocauc hard {} (th={})".format(name, roc_hard, th) + "\n")
    print('{} coverage of rules: {:.2f}%'.format(name, rule_coverage) + "\n")
    print('{} accuracy of rules: {:.2f}%'.format(name, rule_accuracy) + "\n")
    print('{} loss of rules: {:.2f}'.format(name, loss_rules) + "\n")
    print('{} coverage of gate in rules: {:.2f}%'.format(name, gate_coverage_in_rules) + "\n")
    print('{} coverage of gate: {:.2f}%'.format(name, gate_coverage) + "\n")
    print('{} accuracy on instances covered by rule set {}'.format(name, acc_rule_instances) + "\n")
    print('{} num rules {}'.format(name, num_rules) + "\n")

    result_dic = {"L_star": L_star, "L_train": L_train, "L_test": L_test, "acc_test": acc, "acc_test_hard": acc_hard, "roc_test": roc, "roc_test_hard": roc_hard, "rule_coverage": rule_coverage,\
     "rule_accuracy": rule_accuracy, "gate_coverage_in_rules": gate_coverage_in_rules, "gate_coverage": gate_coverage, "acc_rule_instances": acc_rule_instances, 'num_rules':num_rules}

    return result_dic

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--seed', default=0, type=int, metavar='S',
                        help='random seed')
    parser.add_argument('--device', default='cpu', type=str,
                        metavar='d', help='device')
    parser.add_argument('--log-path', default='./logs/', type=str,
                        metavar='lp', help='name of log dir')
    parser.add_argument('--epsilon', default=0.1, type=float, metavar='e',
                        help='epsilon for loss difference')
    parser.add_argument('--lr', default=0.01, type=float, metavar='LR',
                        help='learning rate') 
    parser.add_argument('--increase-epochs', default=0, type=float, metavar='ie',
                        help='epochs in which lr is reached') 
    parser.add_argument('--epochs-unconst', default=1000, type=int, metavar='eu',
                        help='epochs for unconstrained training')
    parser.add_argument('--epochs-const', default=1000, type=int, metavar='ec',
                        help='epochs for constrained training')
    parser.add_argument('--warmup-unconst', default=0, type=int, metavar='w',
                        help='warmup with unconstrained objective epochs')
    parser.add_argument('--freeze-epochs', default=0, type=int, metavar='w',
                        help='number of epochs freezing black-box classifier in constrained optimization')
    parser.add_argument('--batch-size', default=32, type=int, metavar='BS',
                        help='batch size')
    parser.add_argument('--alpha', default=1, type=float, metavar='a',
                        help='alpha in dbgd')
    parser.add_argument('--beta', default=1, type=float, metavar='b',
                        help='beta in dbgd')
    parser.add_argument('--gamma', default=1.0, type=float, metavar='g',
                        help='regularization weight')
    parser.add_argument('--tau-norm', default=2.0, type=float, metavar='t',
                        help='norm for phi in dbgd')
    parser.add_argument('--lex-norm', default=2.0, type=float, metavar='t',
                        help='norm for lexicographic objective in dbgd')
    parser.add_argument('--norm-grads', default=1.0, type=float, metavar='t',
                        help='normalize gradients')
    parser.add_argument('--dataset', default='adult', type=str, metavar='ds',
                        help='name of training datasets')
    parser.add_argument('--gate', default="nn", type=str, metavar='g',
                        help='gate "linear" or  "nn"')
    parser.add_argument('--balance-dataset', default=0, type=int, metavar='rml',
                        help='balance the used data set')
    parser.add_argument('--max-rules', default=50, type=int, metavar='mr',
                        help='maximum number of rules in final model')
    parser.add_argument('--max-steps', default=5, type=int, metavar='ms',
                        help='maximum number of steps')
    parser.add_argument('--with-llm', default=1, type=int, metavar='wllm',
                        help='use llm for pruning and adaptation')
    parser.add_argument('--seperate-backward', default=1, type=int, metavar='sb',
                        help='seperate backward pass for interpretability loss')
    parser.add_argument('--early-stopping', default=1, type=int, metavar='es',
                        help='include early stopping for unconstrained step')
    parser.add_argument('--load-unconst', default=0, type=int, metavar='luc',
                        help='load pretrained unconstrained model')
    parser.add_argument('--model', default="mlp", type=str, metavar='m',
                        help='specifiy model type [tab_model, mlp or linear]')
    parser.add_argument('--hidden-dim', default=50, type=int, metavar='hd',
                        help='hidden dim for mlp')
    parser.add_argument('--num-rules', default=8, type=int, metavar='nr',
                        help='number of rules to generate in each iteration')
    parser.add_argument('--linear', default=0, type=int, metavar='nr',
                        help='use linear model')
    parser.add_argument('--anchor-th', default=0.95, type=float, metavar='ath',
                        help='fidelity threshold for anchors method')
    parser.add_argument('--unconst-loss', default=0, type=int, metavar='cl',
                        help='train without constraint')
    parser.add_argument('--phi-setting', default="both", type=str, metavar='ol',
                        help='lex --> only lexicographic optimization, const --> constrained with epsilon slack, both')
    parser.add_argument('--rule-strategy', default="mix", type=str, metavar='rs',
                        help='rule strategy: explore, exploit, mix')

    args = parser.parse_args()

    set_seed(args.seed)

    print(args)

    l1_norm = lambda x: sum(p.abs().sum() for p in x.parameters())

    name_datasets = [args.dataset]
    loss_task_func = nn.CrossEntropyLoss()

    for name_dataset in name_datasets:
        device = args.device
        seed = args.seed
        set_seed(seed)
        if not os.path.isdir(args.log_path):
            os.mkdir(args.log_path)
        folder_name = args.log_path + name_dataset + "_" + datetime.datetime.now().strftime("%Y%m%d_T%H%M%S")

        writer = SummaryWriter(folder_name)

        with open(folder_name + "/args.json", 'w') as f:
            json.dump(vars(args), f)
        rule_file = open(folder_name + '/rules.txt'.format(name_dataset), 'w+')
        (X_train, X_test, y_train, y_test), meta_dict = load_data(dataset=args.dataset, split=0.33, save=True, balance=args.balance_dataset)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True, random_state=args.seed)

        ordinal_features = [x for x in range(len(meta_dict["feature_names"])) if x not in list(meta_dict["category_map"].keys())]
        ordinal_transformer = Pipeline(steps=[('scaler', StandardScaler())])

        categorical_features = list(meta_dict["category_map"].keys())
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

        if len(categorical_features):
            preprocessor = InvertableColumnTransformer(transformers=[('num', ordinal_transformer, ordinal_features),
                                                        ('cat', categorical_transformer, categorical_features)], sparse_threshold=0)
        else:
            preprocessor = InvertableColumnTransformer(transformers=[('num', ordinal_transformer, ordinal_features)], sparse_threshold=0)
        preprocessor.fit(X_train)

        if name_dataset == 'adult':
            X_train, y_train = torch.from_numpy(preprocessor.transform(X_train)).to(args.device).float(), torch.from_numpy(y_train).to(args.device).float()
            X_test, y_test = torch.from_numpy(preprocessor.transform(X_test)).to(args.device).float(), torch.from_numpy(y_test).to(args.device).float()
            X_val, y_val = torch.from_numpy(preprocessor.transform(X_val)).to(args.device).float(), torch.from_numpy(y_val).to(args.device).float()
        else:
            X_train, y_train = torch.from_numpy(preprocessor.transform(X_train)).to(args.device).float(), torch.from_numpy(y_train).to(args.device).float()
            X_test, y_test = torch.from_numpy(preprocessor.transform(X_test)).to(args.device).float(), torch.from_numpy(y_test).to(args.device).float()
            X_val, y_val = torch.from_numpy(preprocessor.transform(X_val)).to(args.device).float(), torch.from_numpy(y_val).to(args.device).float()

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=X_test.shape[0])
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=X_val.shape[0])

        if args.model == "tab_model":
            tab_model_c = TabModel(meta_dict["input_dim"], output_dim=2)
            tab_model_g = TabModel(meta_dict["input_dim"], output_dim=2)
            ml_model = MLExpert(input_dim=meta_dict["input_dim"], predef_model=tab_model_c).to(args.device)
            gate = Gate_multirules(input_dim=meta_dict["input_dim"], output_dim=meta_dict["output_dim"], predef_model=tab_model_g)
        elif args.model == "mix":
            ml_model = MLExpert(input_dim=meta_dict["input_dim"], hidden_dim=args.hidden_dim, linear=0).to(args.device)
            gate = Gate_multirules(input_dim=meta_dict["input_dim"], output_dim=meta_dict["output_dim"], hidden_dim=args.hidden_dim, linear=1)
        else:
            ml_model = MLExpert(input_dim=meta_dict["input_dim"], hidden_dim=args.hidden_dim, linear=args.linear).to(args.device)
            gate = Gate_multirules(input_dim=meta_dict["input_dim"], output_dim=meta_dict["output_dim"], hidden_dim=args.hidden_dim, linear=args.linear)
        crs = ClassificationRuleSet(meta_dict, args, reorder=True, use_llm=True, transform=preprocessor, prompt_file='./prompts/{}.json'.format(name_dataset))
        moe = MoE_multirules(gate, ml_model, crs, transform=preprocessor.inverse_transform)	
        loss_task_func = nn.CrossEntropyLoss()
        if args.early_stopping:
            es_kwargs = {'patience':30, 'min_delta':0.0001, 'save':True, 'name':name_dataset + args.model + str(args.hidden_dim), 'path':'./models/'}
        if args.load_unconst:
            unconstrained_path = os.path.join(args.log_path, 'unconstrained')
            moe.load(unconstrained_path)
        else:
            if not args.early_stopping:
                es_kwargs = None
            train_unconstrained(moe, train_loader, val_loader, args, output_len=3, writer=writer, es_kwargs=es_kwargs)
            step_path = os.path.join(folder_name, 'Unconstrained')
            if not os.path.isdir(step_path):
                os.mkdir(step_path)
            moe.save(step_path)
            with torch.no_grad():
                L_star = loss_task_func(moe(X_train)[0].squeeze(), y_train.long())
                L_test = (loss_task_func(moe(X_test)[0].squeeze(), y_test.long())).detach().cpu().numpy()
                L_train = (loss_task_func(moe(X_train)[0].squeeze(), y_train.long())).detach().cpu().numpy()
            res_dic = log_performance(moe, X_test, y_test, L_star, L_star, L_test, "Unconstrained", preprocessor, r_file=rule_file)
            moe.rule_expert.gen_new_rules(moe.ml_model, X_train, y_train, strategy=args.rule_strategy, num=args.num_rules, anchor_th=args.anchor_th, is_sklearn=False, multi_class=True, seed=args.seed)
            unconstrained_path = os.path.join(args.log_path, 'unconstrained')
            if not os.path.isdir(unconstrained_path):
                os.mkdir(unconstrained_path)
            moe.save(unconstrained_path)
        step = 0
        while (step < args.max_steps) and (len(moe.rule_expert.rules) < args.max_rules):
            if step != 0:
                with torch.no_grad():
                    output = moe.rule_expert(torch.tensor(preprocessor.inverse_transform(X_train)))
                    mask = torch.sum(output,axis=-1).bool()
                moe.rule_expert.gen_new_rules(moe.ml_model, X_train[~mask], y_train[~mask], strategy=args.rule_strategy, num=args.num_rules, anchor_th=args.anchor_th, is_sklearn=False, multi_class=True)
            res_dic = log_performance(moe, X_test, y_test, L_star, L_train, L_test, "Constrained bevor LLM Step {}".format(step), preprocessor, r_file=rule_file)
            if args.with_llm:
                tries = 0
                success = None
                rules_prior = moe.rule_expert.rules
                descriptions_prior = moe.rule_expert.descriptions
                labels_prior = moe.rule_expert.rule_labels
                samples_prior = moe.rule_expert.samples
                while (success is None) and tries < 3:
                    tries += 1
                    try:
                        moe.rule_expert.llm_adapt_rules()
                        success = moe.rule_expert(torch.tensor(preprocessor.inverse_transform(X_train)))
                    except Exception as inst:
                        success = None
                        moe.rule_expert.rules = rules_prior
                        moe.rule_expert.descriptions = descriptions_prior
                        moe.rule_expert.rule_labels = labels_prior
                        moe.rule_expert.samples = samples_prior
                        print("LLM rule adaptation failed ", inst)
                    if len(moe.rule_expert.rules) != len(rules_prior):
                        print("Some rule seems to be removed by LLM.")
                        success = None
                res_dic = log_performance(moe, X_test, y_test, L_star, L_train, L_test, "Constrained after adaptation LLM Step {}".format(step), preprocessor, r_file=rule_file)
                time.sleep(60)
                success = None
                tries = 0
                rules_prior = moe.rule_expert.rules
                descriptions_prior = moe.rule_expert.descriptions
                labels_prior = moe.rule_expert.rule_labels
                samples_prior = moe.rule_expert.samples
                while (success is None) and tries < 3:
                    tries += 1
                    try:
                        moe.rule_expert.llm_prune_rules()
                        success = moe.rule_expert(torch.tensor(preprocessor.inverse_transform(X_train)))
                    except Exception as inst:
                        success = None
                        moe.rule_expert.rules = rules_prior
                        moe.rule_expert.descriptions = descriptions_prior
                        moe.rule_expert.rule_labels = labels_prior
                        moe.rule_expert.samples = samples_prior
                        print("LLM rule pruning failed ", inst)
                res_dic = log_performance(moe, X_test, y_test, L_star, L_train, L_test, "Constrained after pruning LLM step {}".format(step), preprocessor, r_file=rule_file)
            losses = train_constrained(moe, L_star, train_loader, args, writer=writer, iteration=step)
        
            with torch.no_grad():
                L_test = (loss_task_func(moe(X_test)[0].squeeze(), y_test.long())).detach().cpu().numpy()
                L_train = (loss_task_func(moe(X_train)[0].squeeze(), y_train.long())).detach().cpu().numpy()
            res_dic = log_performance(moe, X_test, y_test, L_star, L_train, L_test, "Constrained after training step {}".format(step), preprocessor, r_file=rule_file)

            step_path = os.path.join(folder_name, 'Step_{}'.format(step))
            if not os.path.isdir(step_path):
                os.mkdir(step_path)
            moe.save(step_path)

            step += 1

        final_path = os.path.join(folder_name, 'final')
        if not os.path.isdir(final_path):
            os.mkdir(final_path)
        moe.save(final_path)
        dataset_id = datasets_list.index(args.dataset) if args.dataset in datasets_list else -1
        writer.add_hparams({"seed": args.seed, "epsilon": args.epsilon, 
                        "max_rules": args.max_rules, "lr": args.lr,
                        "increase_epochs": args.increase_epochs, "epochs_unconst": args.epochs_unconst,
                        "epochs_const": args.epochs_const, "warmup_unconst": args.warmup_unconst, "freeze_epochs": args.freeze_epochs, "batch_size": args.batch_size,
                        "norm_grads": args.norm_grads, "tau_norm": args.tau_norm, "lex_norm": args.lex_norm,
                        'dataset':dataset_id, "alpha": args.alpha, "beta": args.beta}, res_dic)
        writer.close()
        rule_file.close()

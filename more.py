"""
Copyright 2024 Siemens AG

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import json
import time
import random
import scipy
import torch
import pickle
import pandas as pd
import numpy as np

from tqdm import trange

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import openai
import tiktoken

from alibi.explainers import AnchorTabular
from alibi.explainers import AnchorTabular
from alibi.utils import gen_category_map

from sklearn.compose import ColumnTransformer

from datasets import fetch_adult, fetch_german, fetch_diabetes
import utils


class InvertableColumnTransformer(ColumnTransformer):
    def inverse_transform(self, X):
        if isinstance(X,pd.DataFrame):
            X = X.to_numpy()
        num_features = 0
        for _, _, indices in self.transformers:
            num_features += len(indices)
        out = np.zeros((X.shape[0], num_features))
        for (name, pipeline, indices), (name_out, indices_out) in zip(self.transformers,self.output_indices_.items()):
            transformer = self.named_transformers_.get(name_out, None)
            arr = X[:, indices_out.start: indices_out.stop]

            out[:,indices] = transformer.inverse_transform(arr)


        return out


class EarlyStopper:
    def __init__(self, patience=5, min_delta=0, save=False, name='My', path='models/' ):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.save=save
        self.name=name
        self.path=path
        

    def __call__(self, validation_loss, model=None):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            if self.save:
                if os.path.isfile(self.path+self.name+'_best.pth'):
                    os.remove(self.path+self.name+'_best.pth')
                torch.save(model.state_dict(), self.path+self.name+'_best.pth')

        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        
        return False


def balance_dataset(X, y):  
    unique_classes, class_counts = np.unique(y, return_counts=True)  
    minority_class = unique_classes[np.argmin(class_counts)]  
    majority_class = unique_classes[np.argmax(class_counts)]  
    majority_indices = np.where(y == majority_class)[0]  
    minority_indices = np.where(y == minority_class)[0]  
    majority_indices_subsampled = np.random.choice(majority_indices, size=len(minority_indices), replace=False)  
    balanced_indices = np.concatenate((minority_indices, majority_indices_subsampled))  
    X_balanced = X[balanced_indices]  
    y_balanced = y[balanced_indices]  
    shuffle_indices = np.random.permutation(len(y_balanced))  
    X_balanced = X_balanced[shuffle_indices]  
    y_balanced = y_balanced[shuffle_indices]  
  
    return X_balanced, y_balanced


def load_data(dataset='adult', split=0.33, save=False, balance=False):
    if dataset == 'adult':
        if save:
            adult = fetch_adult()
            with open('adult.pkl', 'wb') as f:
                pickle.dump(adult, f)
        else:
            with open('adult.pkl', 'rb') as f:
                adult = pickle.load(f)
        data = adult.data
        target = adult.target
        if balance:
            data, target = balance_dataset(data, target)
        feature_names = adult.feature_names
        category_map = adult.category_map
        class_names = adult.target_names
        meta_dict = {"feature_names": feature_names, "class_names": class_names, "category_map": category_map, "input_dim": 54, "output_dim":2}
        ds = train_test_split(data, target, test_size=split, random_state=0, shuffle=True)
        return ds, meta_dict
    if dataset == 'german':
        if save:
            adult = fetch_german()
            with open('german.pkl', 'wb') as f:
                pickle.dump(adult, f)
        else:
            with open('german.pkl', 'rb') as f:
                adult = pickle.load(f)
        data = adult.data
        target = adult.target
        if balance:
            data, target = balance_dataset(data, target)
        feature_names = adult.feature_names
        category_map = adult.category_map
        class_names = adult.target_names
        meta_dict = {"feature_names": feature_names, "class_names": class_names, "category_map": category_map, "input_dim": 61, "output_dim":2}
        ds = train_test_split(data, target, test_size=split, random_state=0, shuffle=True)
        return ds, meta_dict
    if dataset == 'diabetes':
        if save:
            adult = fetch_diabetes()
            with open('diabetes.pkl', 'wb') as f:
                pickle.dump(adult, f)
        else:
            with open('diabetes.pkl', 'rb') as f:
                adult = pickle.load(f)
        data = adult.data
        target = adult.target
        if balance:
            data, target = balance_dataset(data, target)
        feature_names = adult.feature_names
        category_map = adult.category_map
        class_names = adult.target_names
        meta_dict = {"feature_names": feature_names, "class_names": class_names, "category_map": category_map, "input_dim": 8, "output_dim":2}
        ds = train_test_split(data, target, test_size=split, random_state=0, shuffle=True)
        return ds, meta_dict
    else:
        print("No such dataset")


def build_prompt():
    cat_feature_map = {}
    for k in adult.category_map.keys():
        cat_feature_map[feature_names[k]] = adult.category_map[k]
    smstring = ''
    for k in cat_feature_map.keys():
        smstring += "{}: {}".format(k, cat_feature_map[k]) + '\n'


def train_unconstrained(model, train_loader, val_loader, args, loss_func="CE", output_len=3, name="unconstrained", writer=None, es_kwargs=None):
    loss_task_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr*10)
    if es_kwargs != None:
        Es = EarlyStopper(**es_kwargs)
    losses = []
    for epoch in trange(1, args.epochs_unconst+1):
        model.train()
        loss_batch = 0
        count = 0
        for batch_train_x, batch_train_y in train_loader:
            optimizer.zero_grad()
            if output_len == 3:
                output, _, _ = model(batch_train_x)
            else:
                output = model(batch_train_x)
            loss = loss_task_func(output, batch_train_y.long())
            loss_batch += loss.item()
            loss.backward()
            optimizer.step()
            count += 1
        losses.append(loss_batch/count)
        model.eval()
        if writer:
            writer.add_scalar('Train/{}/loss/task_loss'.format(name) , loss_batch/count, epoch)
        with torch.no_grad():
            for x, y in val_loader:
                output, _, _ = model(x)
                loss = loss_task_func(output, y.long())
        v_loss = loss.mean().item()
        writer.add_scalar('Validation/{}/loss/task_loss'.format(name) ,v_loss, epoch)
        if es_kwargs is not None:
            if Es(v_loss, model=model):
                print(f'Early Stopping at epoch {epoch}')
                model.load_state_dict(torch.load(es_kwargs['path']+es_kwargs['name']+'_best.pth'))
                break


def train_constrained(moe, L_s, train_loader, args, loss_func="CE", use_batch=True, iteration=0, writer=None, verbose=0):
    loss_task_func = nn.CrossEntropyLoss()

    optimizer = optim.SGD(moe.ml_model.parameters(), lr=args.lr)

    param_shapes_moe = [param.size() for param in moe.parameters()]
    param_shapes_ml = [param.size() for param in moe.ml_model.parameters()]
    param_shapes_gate = [param.size() for param in moe.gating_model.parameters()]

    if args.freeze_epochs > 0:
        freeze_model(moe.ml_model)
    losses = []
    for epoch in trange(1, args.epochs_const+1):
        moe.train()
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs:
            unfreeze_model(moe.ml_model)

        # learning rate schedule
        if not args.increase_epochs:
            lr = args.lr
        else:
            if epoch < args.increase_epochs:
                lr = args.lr * (epoch/args.increase_epochs)

        metrics_dict = {'lambda':0, 'lr':0, 'gamma':0, 'phi':0, 'dot':0,
                'gate_grad_task':0, 'ml_grad_task':0, 'gate_grad_int':0, 'ml_grad_int':0,
                'const':0, 'grad_task':0, 'grad_int':0, 
                'task_loss':0, 'norm':0, 'interpretability_loss':0, 'phi_bin': 0}

        # training loop
        count = 0
        for batch_train_x, batch_train_y in train_loader:

            if args.unconst_loss:
                output, gate, mask = moe(batch_train_x)

                g = loss_task_func(output, batch_train_y.long())
                g.backward()
                g_grads = []
                weights = []
                for param in moe.parameters():
                    g_grads.append(param.grad.view(-1))
                    weights.append(param.data.view(-1))
                g_grads = torch.cat(g_grads)
                weights = torch.cat(weights)
                if args.norm_grads:
                    g_grads = g_grads / torch.norm(g_grads,p=2)

                weights = weights - (lr*g_grads)
                weights = utils.reshape_params(weights, param_shapes_moe)
                
                for param, updated_param in zip(moe.parameters(), weights):
                    param.data.copy_(updated_param)
                moe.zero_grad()

                metrics_dict['grad_task'] += torch.norm(g_grads)
                metrics_dict['gate_grad_task'] += torch.norm(g_grads[-utils.size_params(param_shapes_gate):])
                metrics_dict['ml_grad_task'] += torch.norm(g_grads[0:utils.size_params(param_shapes_gate)])
                metrics_dict['task_loss'] += g
            else:
                output, gate, mask = moe(batch_train_x)

                g = loss_task_func(output, batch_train_y.long())
                g.backward()
                g_grads = []
                weights = []
                for param in moe.ml_model.parameters():
                    g_grads.append(param.grad.view(-1))
                    weights.append(param.data.view(-1))
                g_grads = torch.cat(g_grads)
                weights = torch.cat(weights)
                if args.norm_grads:
                    g_grads = g_grads / torch.norm(g_grads,p=2)

                weights = weights - (lr*g_grads)
                weights = utils.reshape_params(weights, param_shapes_ml)
                
                for param, updated_param in zip(moe.ml_model.parameters(), weights):
                    param.data.copy_(updated_param)
                moe.zero_grad()

                metrics_dict['grad_task'] += torch.norm(g_grads)
                metrics_dict['gate_grad_task'] += torch.norm(g_grads[-utils.size_params(param_shapes_gate):])
                metrics_dict['ml_grad_task'] += torch.norm(g_grads[0:utils.size_params(param_shapes_gate)])
                metrics_dict['task_loss'] += g

                if torch.sum(mask) != 0.0:
                    output, gate, mask = moe(batch_train_x)
                    if args.seperate_backward:
                        g = 0
                        for i,m in enumerate(mask):
                            if m:
                                g_ = loss_task_func(output[i].unsqueeze(dim=0), batch_train_y[i].unsqueeze(dim=0).long())
                                g_.backward(retain_graph=True)
                                g += g_.detach().numpy()
                        g /= np.sum(mask.detach().numpy())
                    else:
                        g = loss_task_func(output, batch_train_y.long())
                        g.backward()
                    g_grads = []
                    g_grads_ml = []
                    for param in moe.gating_model.parameters():
                        g_grads.append(param.grad.view(-1))
                    g_grads = torch.cat(g_grads)
                    if args.norm_grads:
                        g_grads = g_grads / torch.norm(g_grads,p=2)
                    
                    moe.zero_grad()

                    # maximization of gate usage as main objective
                    output, gate, mask = moe(batch_train_x)

                    if args.seperate_backward:
                        f = 0
                        for i,m in enumerate(mask):
                            if m:
                                f_ = -torch.log(gate[i,1])
                                f_.backward(retain_graph=True)
                                f += f_.detach().numpy()
                        f /= np.sum(mask.detach().numpy())
                    else:
                        f = torch.mean(-torch.log(gate[:,1]))
                        f.backward()

                    f_grads = []
                    weights = []
                    for param in moe.gating_model.parameters():
                        f_grads.append(param.grad.view(-1))
                        weights.append(param.data.view(-1))

                    f_grads = torch.cat(f_grads)
                    if args.norm_grads:
                        f_grads = f_grads / torch.norm(f_grads,p=2)
                    weights = torch.cat(weights)

                    # constrained definition
                    const = args.alpha*(g - (1 + args.epsilon)*L_s)

                    # use different norms for phi and lambda
                    dot = torch.dot(f_grads, g_grads)
                    if args.tau_norm:
                        norm = args.beta*((torch.norm(g_grads, p=args.lex_norm)))
                        if args.phi_setting == "lex":
                            phi = norm
                        elif args.phi_setting == "const":
                            phi = const
                        else:
                            phi = torch.min(torch.tensor((const, norm)))
                        lambda__ = (phi - dot)/(torch.norm(g_grads,p=args.tau_norm))
                        lambda_ = torch.max(torch.tensor((lambda__, 0.0)))
                    else:
                        norm = args.beta*((torch.norm(g_grads, p=2)**2))
                        phi = torch.min(torch.tensor((const, norm)))
                        lambda__ = (phi - dot)/(torch.norm(g_grads,p=2)**2)
                        lambda_ = torch.max(torch.tensor((lambda__, 0.0)))
                    
                    # use different gammas as multiplicative factor for gamma
                    if not args.gamma:
                        # adaptive gamma only makes sense if gradients are not normalized
                        gamma_ = (torch.norm(f_grads,p=2)/torch.norm(g_grads,p=2))
                        if gamma_ < 1:
                            gamma_ = 1
                    else:
                        gamma_ = args.gamma

                    if epoch < args.warmup_unconst:
                        grads = g_grads
                    else:
                        grads = f_grads + gamma_ * lambda_ * g_grads
                    weights = weights - (lr*grads)
                    weights = utils.reshape_params(weights, param_shapes_gate)

                    
                    for param, updated_param in zip(moe.gating_model.parameters(), weights):
                        param.data.copy_(updated_param)
                    moe.zero_grad()
                
                    metrics_dict['lambda'] += lambda_
                    metrics_dict['lr'] += lr
                    metrics_dict['gamma'] += gamma_
                    metrics_dict['phi'] += phi
                    metrics_dict['dot'] += dot
                    metrics_dict['const'] += const/args.alpha
                    metrics_dict['grad_task'] += torch.norm(g_grads)
                    metrics_dict['norm'] += norm/args.beta
                    metrics_dict['grad_int'] += torch.norm(f_grads)
                    metrics_dict['task_loss'] += g
                    metrics_dict['interpretability_loss'] += f
                    metrics_dict['phi_bin'] += (1 if const > norm else 0)
            count += 1
        for k in metrics_dict.keys():
            metrics_dict[k] /= count
        if writer:
            writer.add_scalar('Train/constrained{}/params/lambda'.format(iteration) , metrics_dict['lambda'], epoch)
            writer.add_scalar('Train/constrained{}/params/lr'.format(iteration) , metrics_dict['lr'], epoch)
            writer.add_scalar('Train/constrained{}/params/gamma'.format(iteration) , metrics_dict['gamma'], epoch)
            writer.add_scalar('Train/constrained{}/params/phi'.format(iteration) , metrics_dict['phi'], epoch)
            writer.add_scalar('Train/constrained{}/params/dot'.format(iteration) , metrics_dict['dot'], epoch)
            writer.add_scalar('Train/constrained{}/params/const'.format(iteration) , metrics_dict['const'], epoch)
            writer.add_scalar('Train/constrained{}/params/norm'.format(iteration) , metrics_dict['norm'], epoch)
            writer.add_scalar('Train/constrained{}/params/grad_task'.format(iteration) , metrics_dict['grad_task'], epoch)
            writer.add_scalar('Train/constrained{}/params/gate_grad_task'.format(iteration) , metrics_dict['gate_grad_task'], epoch)
            writer.add_scalar('Train/constrained{}/params/ml_grad_task'.format(iteration) , metrics_dict['ml_grad_task'], epoch)
            writer.add_scalar('Train/constrained{}/params/gate_grad_int'.format(iteration) , metrics_dict['gate_grad_int'], epoch)
            writer.add_scalar('Train/constrained{}/params/ml_grad_int'.format(iteration) , metrics_dict['ml_grad_int'], epoch)
            writer.add_scalar('Train/constrained{}/params/grad_int'.format(iteration) , metrics_dict['grad_int'], epoch)
            writer.add_scalar('Train/constrained{}/loss/task_loss'.format(iteration) , metrics_dict['task_loss'], epoch)
            writer.add_scalar('Train/constrained{}/loss/interpretability_loss'.format(iteration) , metrics_dict['interpretability_loss'], epoch)
            writer.add_scalar('Train/constrained{}/loss/phi_bin'.format(iteration) , metrics_dict['phi_bin'], epoch)
        losses.append(metrics_dict['task_loss'])
    return losses


class ClassificationRuleSet(nn.Module):
    def __init__(self, meta_dict, args, transform, reorder=True, use_llm=True, prompt_file='./prompts/adult.json'):
        super(ClassificationRuleSet, self).__init__()
        self.rules = []
        self.descriptions = []
        self.rule_labels = []
        self.samples = []
        self.order = None
        self.idx = None
        self.reorder = reorder
        self.pruning_history = {}
        self.args = args

        self.meta_dict = meta_dict
        self.transform = transform
        self.cat_features = [meta_dict["feature_names"][i] for i in list(meta_dict["category_map"].keys())]

        if use_llm:
            self.client = None #TODO: define your OpenAI API access here

            with open(prompt_file, 'r') as f:
                self.prompt_dict = json.load(f)
                
    def forward(self, x_):
        out = torch.zeros(x_.shape[0], 2)

        if len(x_.shape) == 1:
            x_ = x_[np.newaxis, :]
        if len(self.rules) == 0:
            return out
        else:
            for i,x in enumerate(x_):
                results = []
                if self.reorder:
                    self.calc_order(self.transform.transform(x[np.newaxis,:]).squeeze())
                    rules_ = [self.rules[int(i)-1] for i in self.order]
                    rule_labels_ = [self.rule_labels[int(i)-1] for i in self.order]
                    descriptions_ = [self.descriptions[int(i)-1] for i in self.order]
                else:
                    rules_ = self.rules
                    rule_labels_ = self.rule_labels
                    descriptions_ = self.descriptions       
                for rule in rules_:
                    predicates = rule.split(' AND ')
                    evaluated_predi = []
                    for elements in predicates:
                        for operator in [' < ', ' > ', ' <= ', ' >= ', ' = ', ' != ']:
                            elements = split_(elements, operator)
                        # handle cat features
                        if len(set(elements).intersection(set(self.cat_features))):
                            cat = elements[0]
                            feat = elements[-1]
                            cat_feature_idx =list(self.meta_dict["category_map"].keys())[self.cat_features.index(cat)] 
                            try:
                                res = int(x[cat_feature_idx])
                            except:
                                print("feature {} is nan in {}".format(cat_feature_idx, x))
                                print(elements)
                                continue
                            ref = self.meta_dict['category_map'][cat_feature_idx].index(feat)
                            pred = res == ref
                        else:
                            if len(elements) == 5:
                                cat = elements[2]
                                res = x[self.meta_dict['feature_names'].index(cat)]
                                pred_1 = eval(elements[0] + elements[1] + "{:.3f} ".format(res))
                                pred_2 = eval("{:.3f} ".format(res) + elements[3] + elements[4] )
                                pred = pred_1 & pred_2
                            elif len(elements) == 3:
                                cat = elements[0]
                                res = x[self.meta_dict['feature_names'].index(cat)]
                                if elements[1] == ' = ' or elements[1] == '=':
                                    elements[1] = ' == '
                                pred = eval("{:.3f} ".format(res) + elements[1] + elements[2])
                            else:
                                print("wrong number of elements in: ", elements)
                                exit()
                        evaluated_predi.append(pred)
                    results.append(all(evaluated_predi))
                self.idx = np.where(results)[0]
                if len(self.idx) > 0:
                    if rule_labels_[self.idx[0]].squeeze()[0]:
                        out[i,0] = 1.0
                    else:
                        out[i,1] = 1.0
            return out

    def describe_rule(self, x):
        y = int(np.argmax(self.forward(x).detach().cpu().numpy(), axis=-1))
        if self.reorder:
            rules_ = [self.rules[int(i)-1] for i in self.order]
            rule_labels_ = [self.rule_labels[int(i)-1] for i in self.order]
            descriptions_ = [self.descriptions[int(i)-1] for i in self.order]
        else:
            rules_ = self.rules
            rule_labels_ = self.rule_labels
            descriptions_ = self.descriptions       
        if len(self.idx) > 0:
            return 'Output: ' + str(y) + '\n' + 'Rule: ' + rules_[self.idx[0]] + ' => ' + self.meta_dict["class_names"][y] + '\n' + descriptions_[self.idx[0]] + '\n' + self.print_instance(x.squeeze())
        else:
            return 'Output: ' + str(-1) + 'No rule applicable'

    def ranking_strategy(self, model, input, predictions, strategy='exploit', num=10):
        if strategy == 'exploit':
            entropies = scipy.stats.entropy(predictions, axis=1)
            idxs_uncertain = np.argsort(entropies)[:num]
        elif strategy == 'explore':
            entropies = scipy.stats.entropy(predictions, axis=1)
            idxs_uncertain = np.argsort(-entropies)[:num]
        elif strategy == 'mix':
            entropies = scipy.stats.entropy(predictions, axis=1)
            idxs_uncertain = []
            idxs_uncertain += list(np.argsort(entropies)[:num//2])
            idxs_uncertain += list(np.argsort(-entropies)[:(num-num//2)])
            idxs_uncertain = np.array(idxs_uncertain)
        else:
            print('No such strategy as:', strategy)
        return idxs_uncertain

    def gen_new_rules(self, model, data, labels, num=10, strategy='exploit', anchor_th=0.95, is_sklearn=True, multi_class=False, seed=0):

        data_it = self.transform.inverse_transform(data)

        if is_sklearn:
            predict_fn = lambda x: model.predict_proba(self.transform.transform(x))
            predict_fn_proba = lambda x: model.predict_proba(self.transform.transform(x))
            predictions = model.predict_proba(data)
        else:
            if multi_class:
                predict_fn = lambda x: model(torch.tensor(self.transform.transform(x)).float()).detach().cpu().numpy()
            else:
                predict_fn = lambda x: np.array([(model(torch.tensor(self.transform.transform(x)).float()) > 0.5).squeeze().float().detach().cpu().numpy()])
            predictions = model(data).detach().cpu().numpy()
        explainer = AnchorTabular(predict_fn, self.meta_dict['feature_names'], categorical_names=self.meta_dict['category_map'], seed=seed)
        explainer.fit(data_it, disc_perc=np.arange(0,100,5))

        idxs_uncertain = self.ranking_strategy(model, data, predictions, strategy=strategy, num=num)

        for i in range(num):
            exp = explainer.explain(data_it[idxs_uncertain[i]][np.newaxis,:], threshold=anchor_th)
            print('new_rule: ', exp.data['anchor'])
            if len(exp.data['anchor']) > 0:
                rule = (' AND '.join(exp.data['anchor'])) if len(exp.data['anchor']) > 1 else exp.data['anchor'][0]
                self.rules.append(rule)
                self.samples.append(data[idxs_uncertain[i]])
                label = torch.zeros((1,len(self.meta_dict["class_names"])))
                label[:,int(labels[idxs_uncertain[i]])] = 1.0
                self.rule_labels.append(label)
                self.descriptions.append('Description: Run llm_adapt_rules to generate description!')

        self.remove_duplicate_rules()

    def llm_prune_rules(self, engine='gpt-4', temperature=0.7, seed=None, top_p=0.95, frequency_penalty=0, presence_penalty=0):
        ruleset = ''
        for r, rl in zip(self.rules, self.rule_labels):
            ruleset += 'Rule: ' + r + ' => ' + self.meta_dict["class_names"][np.argmax(rl.detach().cpu().numpy())] + '\n'

        history = []
        prompt = self.prompt_dict['pruning']['prompt'] + '\n' + ruleset
        history.append({"role":"system","content": self.prompt_dict['pruning']['general_prompt']})
        history.append({"role":"user","content": prompt})
        prompt_tokens = num_tokens_from_string(prompt, 'gpt-4')
        general_prompt_tokens = num_tokens_from_string(self.prompt_dict['pruning']['general_prompt'], 'gpt-4')
        max_tokens = 8192 - prompt_tokens - general_prompt_tokens - 100
    
        completion = self.client.chat.completions.create(
            model = engine,
            messages = history,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            stop=None
        )
        
        answer=completion.choices[0].message.content
        print(answer)
        pruning_step = str(len(self.pruning_history.keys()))
        if len(answer.split(' Reasoning')) != 1 and answer.split(' Reasoning')[0] != 'None':
            pruned_rules = eval(answer.split(' Reasoning')[0])
            self.pruning_history[pruning_step] = {}
            self.pruning_history[pruning_step]['Rules'] = [self.rules[i-1] + ' => ' + self.meta_dict["class_names"][np.argmax(self.rule_labels[i-1].detach().cpu().numpy())] for i in pruned_rules]
            self.pruning_history[pruning_step]['Reason'] = answer.split(' Reasoning')[1][2:]
            self.pruning_history[pruning_step]['Removed'] = answer.split(' Reasoning')[0]

            for i in sorted(pruned_rules, reverse=True):
                del self.rules[i-1]
                del self.descriptions[i-1]
                del self.rule_labels[i-1]
                del self.samples[i-1]
        else:
            self.pruning_history[pruning_step] = "No pruning"

        return 0

    def llm_adapt_rules(self, engine='gpt-4', temperature=0.7, seed=None, top_p=0.95, frequency_penalty=0, presence_penalty=0):
        ruleset = ''
        for r, rl in zip(self.rules, self.rule_labels):
            ruleset += 'Rule: ' + r + ' => ' + self.meta_dict["class_names"][np.argmax(rl.detach().cpu().numpy())] + '\n'
        print(ruleset)

        history = []
        prompt = self.prompt_dict['adaptation']['prompt'] + '\n' + ruleset
        history.append({"role":"system","content": self.prompt_dict['adaptation']['general_prompt']})
        history.append({"role":"user","content": prompt})
        prompt_tokens = num_tokens_from_string(prompt, 'gpt-4')
        general_prompt_tokens = num_tokens_from_string(self.prompt_dict['adaptation']['general_prompt'], 'gpt-4')
        max_tokens = 8192 - prompt_tokens - general_prompt_tokens - 100
    
        completion = self.client.chat.completions.create(
            model = engine,
            messages = history,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            stop=None
        )
        
        answer=completion.choices[0].message.content
        print(answer)

        if len(answer.split("Rule: ")[1:]) != len(self.rules):
            print("Some rule seems to be removed by LLM. Previous rules: {}, New rules: {}".format(len(answer.split("Rule: ")[1:]), len(self.rules)))
            print(answer)
            print(answer.split("Rule: "))
            return None
        
        for i,ele in enumerate(answer.split("Rule: ")[1:]):
            self.rules[i] = ele.split('\n')[0].split(' => ')[0]
            if np.argmax(self.rule_labels[i]) != self.meta_dict['class_names'].index(ele.split('\n')[0].split(' => ')[1].replace(' ','')):
                print("Rule {} LLM changed label from {} to {}".format(i,self.meta_dict['class_names'][np.argmax(self.rule_labels[i])], ele.split('\n')[0].split(' => ')[1].replace(' ','')))
                label = torch.zeros((1,len(self.meta_dict["class_names"])))
                label[:,self.meta_dict['class_names'].index(ele.split('\n')[0].split(' => ')[1].replace(' ',''))] = 1.0
                self.rule_labels[i] = label
            self.descriptions[i] = ele.split('\n')[1]
        
        self.remove_duplicate_rules()

        return 0
    
    def calc_order(self, x):
        order = np.arange(np.dstack(self.samples).shape[-1]).squeeze()
        dists = np.sum((np.dstack(self.samples).squeeze().T - x)**2, axis=1) / np.dstack(self.samples).squeeze().shape[0]
        self.order = np.argsort(dists)
    
    def remove_duplicate_rules(self):
        new_rules = []
        new_rule_labels = []
        new_descriptions = []
        new_samples = []
        for i in range(len(self.rules)):
            if self.rules[i] not in new_rules:
                new_rules.append(self.rules[i])
                new_rule_labels.append(self.rule_labels[i])
                new_samples.append(self.samples[i])
                new_descriptions.append(self.descriptions[i])
            else:
                print("Duplicate removed: ", self.rules[i])
        self.rules = new_rules
        self.rule_labels = new_rule_labels
        self.descriptions = new_descriptions
        self.samples = new_samples

    def print_instance(self, x):
        out = ''
        for i,k in enumerate(self.meta_dict['feature_names']):
            if i in list(self.meta_dict['category_map'].keys()):
                out += '{}: {}; '.format(k, self.meta_dict['category_map'][i][int(x[i])])
            else:
                out += '{}: {:.2f}; '.format(k, x[i])
        return out


def calculate_coverage_of_ruleset(x):
    return np.sum(np.sum(np.array(x),axis=-1) > 0) / len(x)


def calculate_accuracy_of_ruleset(x, y):
    covered_idx = np.sum(np.array(x),axis=-1) > 0
    return (torch.sum(np.argmax(x[covered_idx], axis=-1) == y[covered_idx]) / np.sum(covered_idx)).item()


def num_tokens_from_string(string, encoding_name):
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def split_(l,d):
    r = []
    if type(l) == list:
        for e in l:
            if d in e:
                r += [e.split(d)[0],d, e.split(d)[1]]
            else:
                r.append(e)
    else:
        if d in l:
            r += [l.split(d)[0],d, l.split(d)[1]]
        else:
            r = l
    return r

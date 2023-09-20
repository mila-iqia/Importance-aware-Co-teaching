from grad import *
from utils import *
import design_bench
import argparse
import higher


# Unpacked Co-teaching Loss function
def loss_coteaching(y_1, y_2, t, num_remember):
    # ind, noise_or_not
    loss_1 = F.mse_loss(y_1, t, reduction='none').view(128)
    ind_1_sorted = np.argsort(loss_1.cpu().data).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.mse_loss(y_2, t, reduction='none').view(128)
    ind_2_sorted = np.argsort(loss_2.cpu().data).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]
    # exchange
    loss_1_update = F.mse_loss(y_1[ind_2_update], t[ind_2_update], reduction='none')
    loss_2_update = F.mse_loss(y_2[ind_1_update], t[ind_1_update], reduction='none')

    return loss_1_update, loss_2_update


def meta_weight(args):
    task = design_bench.make(args.task)
    load_y(args.task)
    task_y0 = task.y
    task_x, task_y, shape0 = process_data(task, args.task, task_y0)
    task_x = torch.Tensor(task_x).to(device)
    task_y = torch.Tensor(task_y).to(device)

    indexs = torch.argsort(task_y.squeeze())
    # Find the x with the maximum corresponding y, that is the top1
    index = indexs[-1:]
    x_init = copy.deepcopy(task_x[index])
    # get top k candidates
    if args.reweight_mode == "top128":
        index_val = indexs[-args.topk:]
    elif args.reweight_mode == "half":
        index_val = indexs[-(len(indexs) // 2):]
    else:
        index_val = indexs
    x_val = copy.deepcopy(task_x[index_val])
    label_val = copy.deepcopy(task_y[index_val])
    f1 = SimpleMLP(task_x.shape[1]).to(device)
    f1.load_state_dict(
        torch.load(args.proxy_path + args.task + "_proxy_" + str(args.seed1) + ".pt", map_location='cuda:0'))
    f2 = SimpleMLP(task_x.shape[1]).to(device)
    f2.load_state_dict(
        torch.load(args.proxy_path + args.task + "_proxy_" + str(args.seed2) + ".pt", map_location='cuda:0'))
    f3 = SimpleMLP(task_x.shape[1]).to(device)
    f3.load_state_dict(
        torch.load(args.proxy_path + args.task + "_proxy_" + str(args.seed3) + ".pt", map_location='cuda:0'))

    candidate = x_init[0]  # i.e., x_0
    candidate.requires_grad = True
    candidate_opt = optim.Adam([candidate], lr=args.ft_lr)
    optimizer1 = torch.optim.Adam(f1.parameters(), lr=args.alpha, weight_decay=args.wd)
    optimizer2 = torch.optim.Adam(f2.parameters(), lr=args.alpha, weight_decay=args.wd)
    optimizer3 = torch.optim.Adam(f3.parameters(), lr=args.alpha, weight_decay=args.wd)
    for i in range(1, args.Tmax + 1):
        loss = -1.0 / 3.0 * (f1(candidate) + f2(candidate) + f3(candidate))
        candidate_opt.zero_grad()
        loss.backward()
        candidate_opt.step()
        x_train = []
        y1_label = []
        y2_label = []
        y3_label = []
        # sample K points around current candidate
        for k in range(args.K):
            temp_x = candidate.data + args.noise_coefficient * np.random.normal(args.mu,
                                                                                args.std)  # add gaussian noise
            x_train.append(temp_x)
            temp_y1 = f1(temp_x)
            y1_label.append(temp_y1)

            temp_y2 = f2(temp_x)
            y2_label.append(temp_y2)

            temp_y3 = f3(temp_x)
            y3_label.append(temp_y3)

        x_train = torch.stack(x_train)
        y1_label = torch.Tensor(y1_label).to(device)
        y1_label = torch.reshape(y1_label, (args.K, 1))
        y2_label = torch.Tensor(y2_label).to(device)
        y2_label = torch.reshape(y2_label, (args.K, 1))
        y3_label = torch.Tensor(y3_label).to(device)
        y3_label = torch.reshape(y3_label, (args.K, 1))

        if args.if_reweight and args.if_coteach:

            # Round 1, use f3 to update f1 and f2
            weight_1 = torch.ones(args.num_coteaching).to(device)
            weight_1.requires_grad = True
            weight_2 = torch.ones(args.num_coteaching).to(device)
            weight_2.requires_grad = True

            with higher.innerloop_ctx(f1, optimizer1) as (model1, opt1):
                with higher.innerloop_ctx(f2, optimizer2) as (model2, opt2):
                    l1, l2 = loss_coteaching(model1(x_train), model2(x_train), y3_label, args.num_coteaching)

                    l1_t = weight_1 * l1
                    l1_t = torch.sum(l1_t) / args.num_coteaching
                    opt1.step(l1_t)

                    logit1 = model1(x_val)
                    loss1_v = F.mse_loss(logit1, label_val)
                    g1 = torch.autograd.grad(loss1_v, weight_1)[0].data
                    g1 = F.normalize(g1, p=args.clamp_norm, dim=0)
                    g1 = torch.clamp(g1, min=args.clamp_min, max=args.clamp_max)
                    weight_1 = weight_1 - args.beta * g1
                    weight_1 = torch.clamp(weight_1, min=0, max=2)

                    l2_t = weight_2 * l2
                    l2_t = torch.sum(l2_t) / args.num_coteaching
                    opt2.step(l2_t)

                    logit2 = model2(x_val)
                    loss2_v = F.mse_loss(logit2, label_val)
                    g2 = torch.autograd.grad(loss2_v, weight_2)[0].data
                    g2 = F.normalize(g2, p=args.clamp_norm, dim=0)
                    g2 = torch.clamp(g2, min=args.clamp_min, max=args.clamp_max)
                    weight_2 = weight_2 - args.beta * g2
                    weight_2 = torch.clamp(weight_2, min=0, max=2)

            loss1 = weight_1 * F.mse_loss(f1(x_train), y3_label, reduction='none')
            loss1 = torch.sum(loss1) / args.num_coteaching
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            loss2 = weight_2 * F.mse_loss(f2(x_train), y3_label, reduction='none')
            loss2 = torch.sum(loss2) / args.num_coteaching
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            # Round 2, use f2 to update f1 and f3
            weight_1 = torch.ones(args.num_coteaching).to(device)
            weight_1.requires_grad = True
            weight_3 = torch.ones(args.num_coteaching).to(device)
            weight_3.requires_grad = True

            with higher.innerloop_ctx(f1, optimizer1) as (model1, opt1):
                with higher.innerloop_ctx(f3, optimizer3) as (model3, opt3):
                    l1, l3 = loss_coteaching(model1(x_train), model3(x_train), y2_label, args.num_coteaching)

                    l1_t = weight_1 * l1
                    l1_t = torch.sum(l1_t) / args.num_coteaching
                    opt1.step(l1_t)

                    logit1 = model1(x_val)
                    loss1_v = F.mse_loss(logit1, label_val)
                    g1 = torch.autograd.grad(loss1_v, weight_1)[0].data
                    g1 = F.normalize(g1, p=args.clamp_norm, dim=0)
                    g1 = torch.clamp(g1, min=args.clamp_min, max=args.clamp_max)
                    weight_1 = weight_1 - args.beta * g1
                    weight_1 = torch.clamp(weight_1, min=0, max=2)

                    l3_t = weight_3 * l3
                    l3_t = torch.sum(l3_t) / args.num_coteaching
                    opt3.step(l3_t)

                    logit3 = model3(x_val)
                    loss3_v = F.mse_loss(logit3, label_val)
                    g3 = torch.autograd.grad(loss3_v, weight_3)[0].data
                    g3 = F.normalize(g3, p=args.clamp_norm, dim=0)
                    g3 = torch.clamp(g3, min=args.clamp_min, max=args.clamp_max)
                    weight_3 = weight_3 - args.beta * g3
                    weight_3 = torch.clamp(weight_3, min=0, max=2)

            loss1 = weight_1 * F.mse_loss(f1(x_train), y2_label, reduction='none')
            loss1 = torch.sum(loss1) / args.num_coteaching
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            loss3 = weight_3 * F.mse_loss(f3(x_train), y2_label, reduction='none')
            loss3 = torch.sum(loss3) / args.num_coteaching
            optimizer3.zero_grad()
            loss3.backward()
            optimizer3.step()

            # Round 3, use f1 to update f2 and f3
            weight_2 = torch.ones(args.num_coteaching).to(device)
            weight_2.requires_grad = True
            weight_3 = torch.ones(args.num_coteaching).to(device)
            weight_3.requires_grad = True
            with higher.innerloop_ctx(f2, optimizer2) as (model2, opt2):
                with higher.innerloop_ctx(f3, optimizer3) as (model3, opt3):
                    l2, l3 = loss_coteaching(model2(x_train), model3(x_train), y1_label, args.num_coteaching)

                    l2_t = weight_2 * l2
                    l2_t = torch.sum(l2_t) / args.num_coteaching
                    opt2.step(l2_t)

                    logit2 = model2(x_val)
                    loss2_v = F.mse_loss(logit2, label_val)
                    g2 = torch.autograd.grad(loss2_v, weight_2)[0].data
                    g2 = F.normalize(g2, p=args.clamp_norm, dim=0)
                    g2 = torch.clamp(g2, min=args.clamp_min, max=args.clamp_max)
                    weight_2 = weight_2 - args.beta * g2
                    weight_2 = torch.clamp(weight_2, min=0, max=2)

                    l3_t = weight_3 * l3
                    l3_t = torch.sum(l3_t) / args.num_coteaching
                    opt3.step(l3_t)

                    logit3 = model3(x_val)
                    loss3_v = F.mse_loss(logit3, label_val)
                    g3 = torch.autograd.grad(loss3_v, weight_3)[0].data
                    g3 = F.normalize(g3, p=args.clamp_norm, dim=0)
                    g3 = torch.clamp(g3, min=args.clamp_min, max=args.clamp_max)
                    weight_3 = weight_3 - args.beta * g3
                    weight_3 = torch.clamp(weight_3, min=0, max=2)

            loss2 = weight_2 * F.mse_loss(f2(x_train), y1_label, reduction='none')
            loss2 = torch.sum(loss2) / args.num_coteaching
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            loss3 = weight_3 * F.mse_loss(f3(x_train), y1_label, reduction='none')
            loss3 = torch.sum(loss3) / args.num_coteaching
            optimizer3.zero_grad()
            loss3.backward()
            optimizer3.step()

        elif args.if_reweight and not args.if_coteach:

            # Round 1, use f3 to update f1 and f2
            weight_1 = torch.ones(args.K).to(device)
            weight_1.requires_grad = True
            weight_2 = torch.ones(args.K).to(device)
            weight_2.requires_grad = True

            with higher.innerloop_ctx(f1, optimizer1) as (model1, opt1):
                with higher.innerloop_ctx(f2, optimizer2) as (model2, opt2):
                    l1 = F.mse_loss(model1(x_train), y3_label, reduction='none')
                    l2 = F.mse_loss(model2(x_train), y3_label, reduction='none')

                    l1_t = weight_1 * l1
                    l1_t = torch.sum(l1_t) / args.K
                    opt1.step(l1_t)

                    logit1 = model1(x_val)
                    loss1_v = F.mse_loss(logit1, label_val)
                    g1 = torch.autograd.grad(loss1_v, weight_1)[0].data
                    g1 = F.normalize(g1, p=args.clamp_norm)
                    g1 = torch.clamp(g1, min=args.clamp_min, max=args.clamp_max)
                    weight_1 = weight_1 - args.beta * g1
                    weight_1 = torch.clamp(weight_1, min=0, max=2)

                    l2_t = weight_2 * l2
                    l2_t = torch.sum(l2_t) / args.K
                    opt2.step(l2_t)

                    logit2 = model2(x_val)
                    loss2_v = F.mse_loss(logit2, label_val)
                    g2 = torch.autograd.grad(loss2_v, weight_2)[0].data
                    g2 = F.normalize(g2, p=args.clamp_norm)
                    g2 = torch.clamp(g2, min=args.clamp_min, max=args.clamp_max)
                    weight_2 = weight_2 - args.beta * g2
                    weight_2 = torch.clamp(weight_2, min=0, max=2)

            loss1 = weight_1 * F.mse_loss(f1(x_train), y3_label, reduction='none')
            loss1 = torch.sum(loss1) / args.K
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            loss2 = weight_2 * F.mse_loss(f2(x_train), y3_label, reduction='none')
            loss2 = torch.sum(loss2) / args.K
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            # Round 2, use f2 to update f1 and f3
            weight_1 = torch.ones(args.K).to(device)
            weight_1.requires_grad = True
            weight_3 = torch.ones(args.K).to(device)
            weight_3.requires_grad = True

            with higher.innerloop_ctx(f1, optimizer1) as (model1, opt1):
                with higher.innerloop_ctx(f3, optimizer3) as (model3, opt3):
                    l1 = F.mse_loss(model1(x_train), y2_label, reduction='none')
                    l3 = F.mse_loss(model3(x_train), y2_label, reduction='none')

                    l1_t = weight_1 * l1
                    l1_t = torch.sum(l1_t) / args.K
                    opt1.step(l1_t)

                    logit1 = model1(x_val)
                    loss1_v = F.mse_loss(logit1, label_val)
                    g1 = torch.autograd.grad(loss1_v, weight_1)[0].data
                    g1 = F.normalize(g1, p=args.clamp_norm)
                    g1 = torch.clamp(g1, min=args.clamp_min, max=args.clamp_max)
                    weight_1 = weight_1 - args.beta * g1
                    weight_1 = torch.clamp(weight_1, min=0, max=2)

                    l3_t = weight_3 * l3
                    l3_t = torch.sum(l3_t) / args.K
                    opt3.step(l3_t)

                    logit3 = model3(x_val)
                    loss3_v = F.mse_loss(logit3, label_val)
                    g3 = torch.autograd.grad(loss3_v, weight_3)[0].data
                    g3 = F.normalize(g3, p=args.clamp_norm)
                    g3 = torch.clamp(g3, min=args.clamp_min, max=args.clamp_max)
                    weight_3 = weight_3 - args.beta * g3
                    weight_3 = torch.clamp(weight_3, min=0, max=2)

            loss1 = weight_1 * F.mse_loss(f1(x_train), y2_label, reduction='none')
            loss1 = torch.sum(loss1) / args.K
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            loss3 = weight_3 * F.mse_loss(f3(x_train), y2_label, reduction='none')
            loss3 = torch.sum(loss3) / args.K
            optimizer3.zero_grad()
            loss3.backward()
            optimizer3.step()

            # Round 3, use f1 to update f2 and f3
            weight_2 = torch.ones(args.K).to(device)
            weight_2.requires_grad = True
            weight_3 = torch.ones(args.K).to(device)
            weight_3.requires_grad = True
            with higher.innerloop_ctx(f2, optimizer2) as (model2, opt2):
                with higher.innerloop_ctx(f3, optimizer3) as (model3, opt3):
                    l2 = F.mse_loss(model2(x_train), y1_label, reduction='none')
                    l3 = F.mse_loss(model3(x_train), y1_label, reduction='none')

                    l2_t = weight_2 * l2
                    l2_t = torch.sum(l2_t) / args.K
                    opt2.step(l2_t)

                    logit2 = model2(x_val)
                    loss2_v = F.mse_loss(logit2, label_val)
                    g2 = torch.autograd.grad(loss2_v, weight_2)[0].data
                    g2 = F.normalize(g2, p=args.clamp_norm)
                    g2 = torch.clamp(g2, min=args.clamp_min, max=args.clamp_max)
                    weight_2 = weight_2 - args.beta * g2
                    weight_2 = torch.clamp(weight_2, min=0, max=2)

                    l3_t = weight_3 * l3
                    l3_t = torch.sum(l3_t) / args.K
                    opt3.step(l3_t)

                    logit3 = model3(x_val)
                    loss3_v = F.mse_loss(logit3, label_val)
                    g3 = torch.autograd.grad(loss3_v, weight_3)[0].data
                    g3 = F.normalize(g3, p=args.clamp_norm)
                    g3 = torch.clamp(g3, min=args.clamp_min, max=args.clamp_max)
                    weight_3 = weight_3 - args.beta * g3
                    weight_3 = torch.clamp(weight_3, min=0, max=2)

            loss2 = weight_2 * F.mse_loss(f2(x_train), y1_label, reduction='none')
            loss2 = torch.sum(loss2) / args.K
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            loss3 = weight_3 * F.mse_loss(f3(x_train), y1_label, reduction='none')
            loss3 = torch.sum(loss3) / args.K
            optimizer3.zero_grad()
            loss3.backward()
            optimizer3.step()

        elif not args.if_reweight and args.if_coteach:
            # f1 label, f2 and f3 coteaching
            loss_2, loss_3 = loss_coteaching(f2(x_train), f3(x_train), y1_label, args)
            optimizer2.zero_grad()
            loss_2.backward()
            optimizer2.step()
            optimizer3.zero_grad()
            loss_3.backward()
            optimizer3.step()

            # f2 label, f1 and f3 coteaching
            loss_1, loss_33 = loss_coteaching(f1(x_train), f3(x_train), y2_label, args)
            optimizer1.zero_grad()
            loss_1.backward()
            optimizer1.step()
            optimizer3.zero_grad()
            loss_33.backward()
            optimizer3.step()

            # f3 label, f1 and f2 coteaching
            loss_11, loss_22 = loss_coteaching(f1(x_train), f2(x_train), y3_label, args)
            optimizer1.zero_grad()
            loss_11.backward()
            optimizer1.step()
            optimizer2.zero_grad()
            loss_22.backward()
            optimizer2.step()

            # optimizer1.zero_grad()
            # ((loss_1 + loss_11) / 2).backward()
            # optimizer1.step()
            # optimizer2.zero_grad()
            # ((loss_2 + loss_22) / 2).backward()
            # optimizer2.step()
            # optimizer3.zero_grad()
            # ((loss_3 + loss_33) / 2).backward()
            # optimizer3.step()

        elif not args.if_reweight and not args.if_coteach:
            pass

    torch.save(f1.state_dict(), args.reweighting_path + args.task + "_proxy_" + str(args.seed1) + "_"
               + str(args.num_coteaching) + ".pt")
    torch.save(f2.state_dict(), args.reweighting_path + args.task + "_proxy_" + str(args.seed2) + "_"
               + str(args.num_coteaching) + ".pt")
    torch.save(f3.state_dict(), args.reweighting_path + args.task + "_proxy_" + str(args.seed3) + "_"
               + str(args.num_coteaching) + ".pt")


def experiment(args):
    task = [args.task]

    seeds1 = [0, 1, 2, 3, 4, 5, 6, 7]
    seeds2 = [7, 0, 1, 2, 3, 4, 5, 6]
    seeds3 = [6, 7, 0, 1, 2, 3, 4, 5]

    seed = [0, 1, 2, 3, 4, 5, 6, 7]

    # Training Proxy
    args.mode = 'train'
    for sd, s1, s2, s3 in zip(seed, seeds1, seeds2, seeds3):
        print("Current seed is " + str(sd), end="\t")
        set_seed(sd)
        args.seed1 = s1
        args.seed2 = s2
        args.seed3 = s3
        for t in task:
            if t == 'TFBind8-Exact-v0' or t == 'TFBind10-Exact-v0':  # since this is a discrete task
                args.ft_lr = 1e-1
            else:
                args.ft_lr = 1e-3
            print("Current task is " + str(t))
            args.task = t
            print("this is my setting", args)
            meta_weight(args)


def design_opt(args):
    task = design_bench.make(args.task)
    load_y(args.task)
    task_y0 = task.y
    task_x, task_y, shape0 = process_data(task, args.task, task_y0)
    task_x = torch.Tensor(task_x).to(device)
    task_y = torch.Tensor(task_y).to(device)

    indexs = torch.argsort(task_y.squeeze())
    index = indexs[-args.topk:]
    x_init = copy.deepcopy(task_x[index])
    scores = []
    for x_i in range(x_init.shape[0]):
        proxy1 = SimpleMLP(task_x.shape[1]).to(device)
        proxy1.load_state_dict(
            torch.load(args.reweighting_path + args.task + "_proxy_" + str(args.seed1) + "_"
                       + str(args.num_coteaching) + ".pt", map_location='cuda:0'))
        proxy2 = SimpleMLP(task_x.shape[1]).to(device)
        proxy2.load_state_dict(
            torch.load(args.reweighting_path + args.task + "_proxy_" + str(args.seed2) + "_"
                       + str(args.num_coteaching) + ".pt", map_location='cuda:0'))
        proxy3 = SimpleMLP(task_x.shape[1]).to(device)
        proxy3.load_state_dict(
            torch.load(args.reweighting_path + args.task + "_proxy_" + str(args.seed3) + "_"
                       + str(args.num_coteaching) + ".pt", map_location='cuda:0'))
        candidate = x_init[x_i:x_i + 1]
        score_before, _ = evaluate_sample(task, candidate, args.task, shape0)
        candidate.requires_grad = True
        candidate_opt = optim.Adam([candidate], lr=args.ft_lr)
        for i in range(1, args.Tmax + 1):
            loss = -1.0 / 3.0 * (proxy1(candidate) + proxy2(candidate) + proxy3(candidate))
            candidate_opt.zero_grad()
            loss.backward()
            candidate_opt.step()
            if i % args.interval == 0:
                score_after, _ = evaluate_sample(task, candidate.data, args.task, shape0)
                print("candidate {} score before {} score now {}".format(x_i, score_before.squeeze(),
                                                                         score_after.squeeze()))
                scores.append(score_after.squeeze())
        x_init[x_i] = candidate.data
    from statistics import median
    max_score = max(scores)
    median_score = median(scores)
    print("After  max {} median {}\n".format(max_score, median_score))
    return max_score, median_score


def evaluate(args, coteaching: bool):
    task = [args.task]

    # Find the optimal design
    result = {}
    for t in task:
        result[t] = {'max': [], 'median': []}
    args.mode = 'design'

    seeds1 = [0, 1, 2, 3, 4, 5, 6, 7]
    seeds2 = [7, 0, 1, 2, 3, 4, 5, 6]
    seeds3 = [6, 7, 0, 1, 2, 3, 4, 5]

    seed = [0, 1, 2, 3, 4, 5, 6, 7]

    for sd, s1, s2, s3 in zip(seed, seeds1, seeds2, seeds3):
        print("Current seed is " + str(sd), end="\t")
        set_seed(sd)
        args.seed1 = s1
        args.seed2 = s2
        args.seed3 = s3
        for t in task:
            if t == 'TFBind8-Exact-v0' or t == 'TFBind10-Exact-v0' or t == 'CIFARNAS-Exact-v0':  # since this is a discrete task
                args.ft_lr = 1e-1
            else:
                args.ft_lr = 1e-3
            print("Current task is " + str(t))
            args.task = t
            print("this is my setting", args)
            max_score, median_score = design_opt(args)
            result[t]['max'].append(max_score)
            result[t]['median'].append(median_score)

    if coteaching:
        np.save("./meta_weight/fine_tune_results_" + str(args.num_coteaching) + ".npy", result)
    else:
        np.save("./meta_weight/baseline_results.npy", result)


def read_result(path):
    if "baseline" in path:
        print("Results of baseline models")
    else:
        print("Results of fine tuned models")
    result = np.load(path, allow_pickle=True).item()
    for t in result:
        print("Results for task " + t + ":")
        print("\tAverage for max score: " + str(np.mean(result[t]['max'])),
              "\tStd for max score: " + str(np.std(result[t]['max'])))
        print("\tAverage for median score: " + str(np.mean(result[t]['median'])),
              "\tStd for median score: " + str(np.std(result[t]['median'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="pairwise offline")
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--task', choices=['Superconductor-RandomForest-v0', 'HopperController-Exact-v0',
                                           'AntMorphology-Exact-v0', 'DKittyMorphology-Exact-v0', 'TFBind8-Exact-v0',
                                           'TFBind10-Exact-v0', 'CIFARNAS-Exact-v0'],
                        type=str, default='TFBind10-Exact-v0')
    parser.add_argument('--mode', choices=['design', 'train'], type=str, default='train')
    # grad descent to train proxy
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--wd', default=0.0, type=float)
    # grad ascent to obtain design
    parser.add_argument('--Tmax', default=100, type=int)
    parser.add_argument('--ft_lr', default=1e-1, type=float)
    parser.add_argument('--topk', default=128, type=int)
    parser.add_argument('--interval', default=100, type=int)
    parser.add_argument('--proxy_path', default="new_proxies/", type=str)
    parser.add_argument('--reweighting_path', default="meta_weight/", type=str)
    parser.add_argument('--K', default=128, type=int)
    parser.add_argument('--mu', default=0, type=int)
    parser.add_argument('--std', default=1, type=int)
    parser.add_argument('--seed1', default=1, type=int)
    parser.add_argument('--seed2', default=10, type=int)
    parser.add_argument('--seed3', default=100, type=int)
    parser.add_argument('--noise_coefficient', type=float, default=0.1)
    parser.add_argument('--alpha', default=1e-3, type=float)
    parser.add_argument('--beta', default=1e-1, type=float)
    parser.add_argument('--num_coteaching', default=64, type=int)
    parser.add_argument('--if_reweight', default=True, type=bool)
    parser.add_argument('--if_coteach', default=True, type=bool)
    parser.add_argument('--clamp_norm', default=1, type=int)
    parser.add_argument('--clamp_min', default=-0.2, type=float)
    parser.add_argument('--clamp_max', default=0.2, type=float)
    parser.add_argument('--reweight_mode', choices=['top128', 'half', 'full'], type=str, default='half')
    args = parser.parse_args()

    device = torch.device('cuda:' + str(args.device))
    print(device)
    args.num_coteaching = 128
    experiment(args)
    evaluate(args, coteaching=True)
    args.num_coteaching = 64
    experiment(args)
    evaluate(args, coteaching=True)
    args.num_coteaching = 32
    experiment(args)
    evaluate(args, coteaching=True)
    args.num_coteaching = 16
    experiment(args)
    evaluate(args, coteaching=True)
    args.num_coteaching = 8
    experiment(args)
    evaluate(args, coteaching=True)
    print("Num of Co-Teaching 128:")
    read_result("./meta_weight/fine_tune_results_128.npy")
    print("Num of Co-Teaching 64:")
    read_result("./meta_weight/fine_tune_results_64.npy")
    print("Num of Co-Teaching 32:")
    read_result("./meta_weight/fine_tune_results_32.npy")
    print("Num of Co-Teaching 16:")
    read_result("./meta_weight/fine_tune_results_16.npy")
    print("Num of Co-Teaching 8:")
    read_result("./meta_weight/fine_tune_results_8.npy")

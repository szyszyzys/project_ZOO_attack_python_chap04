import random

import torch
# import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import grad

from CNN.resnet import ResNet18
from load_data import load_data

# load the mnist dataset (images are resized into 32 * 32)
training_set, test_set = load_data(data='mnist')

# define the model
model = ResNet18(dim=1)

# detect if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# load the learned model parameters
model.load_state_dict(torch.load('./model_weights/cpu_model.pth'))

model.to(device)
model.eval()

# todo note below is an example of getting the Z(X) vector in the ZOO paper

'''
z = model(image)

# if we consider just one image with size (1, 1, 32, 32)
# z.size() :   (1, 10)  10 elements are corresponding to classes

'''

# attack hyper parameters
attack_hparams = {
    'lr': 0.01,
    'c_search_round': 1000,
    'initial_const': 0.001,
    'max_iteration': 1000,
    'transfer_param': 0.0,
}

'''
calculate the second part of loss function 
in targeted attack: other_logits - class_logits
'''
import random

import torch
# import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import grad


# load the mnist dataset (images are resized into 32 * 32)
training_set, test_set = load_data(data='mnist')

# define the model
model = ResNet18(dim=1)

# detect if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# load the learned model parameters
model.load_state_dict(torch.load('/content/drive/MyDrive/dl/zoo_attack/cpu_model.pth'))

model.to(device)
model.eval()

# todo note below is an example of getting the Z(X) vector in the ZOO paper

'''
z = model(image)

# if we consider just one image with size (1, 1, 32, 32)
# z.size() :   (1, 10)  10 elements are corresponding to classes

'''

# attack hyper parameters
attack_hparams = {
    'lr': 0.01,
    'c_search_round': 10,
    'initial_const': 0.001,
    'max_iteration': 10,
    'transfer_param': 0.0,
}

'''
calculate the second part of loss function 
in targeted attack: other_logits - class_logits
'''
def get_partial_loss(output, labels, labels_infhot):
    # get the confidence score of the target class(real class)
    target_class_logits = output.gather(1, labels.unsqueeze(1)).squeeze(1)
    # get the maximum confidence score of all other classes
    # - labels_infhot to make sure the probability of the target class is -inf.
    other_logits = (output - labels_infhot).amax(dim=1)

    target_class_logits = torch.log(target_class_logits)
    other_logits = torch.log(other_logits)

    return target_class_logits - other_logits


def get_loss(p1, p2):
    return p1 + p2


def zoo_attack(network, images, t_0):
    '''

    #todo you are required to complete this part
    :param network: the model
    :param image: one image with size: (1, 1, 32, 32) type: torch.Tensor()
    :param t_0: real label
    :return: return a torch tensor (attack image) with size (1, 1, 32, 32)
    '''
    # transfer_param: higher value means more tamper.
    transfer_param = attack_hparams['transfer_param']
    c_search_round = attack_hparams['c_search_round']
    attack_iterations = attack_hparams['max_iteration']
    # batch_size = len(image)
    # batch_view = lambda tensor: tensor.view(batch_size, *[1] * (image.ndim - 1))
    t_image = (images * 2).sub_(1).mul_(1 - 1e-6).atanh_()

    # setup regularization parameter c, initialization
    # lower_bound, upper_bound: use binary search to find the optimal c
    c = torch.tensor([attack_hparams['initial_const']], device=device)
    lower_bound = torch.zeros_like(c)
    upper_bound = torch.full_like(c, 1e10)

    # save the global best AE and the l2 distance from the original image
    global_best_l2 = torch.full_like(c, float('inf'))
    global_best_adv = images.clone()
    # if found an adversarial example
    # global_adv_found = torch.zeros(batch_size, device=device, dtype=torch.bool)
    # tools for calculate loss
    label_onehot = None
    label_infhot = None

    for cur_round in range(c_search_round):
        # setup the perturbation_layer and the optimizer, perturbation_layer is the perturbation to the image
        perturbation_layer = torch.zeros_like(images, requires_grad=True)
        optimizer = optim.Adam([perturbation_layer], lr=attack_hparams['lr'])
        # best result with current regularization parameter c
        best_l2 = torch.full_like(c, float('inf'))
        local_adv_found = torch.zeros(1, device=device, dtype=torch.bool)

        # The last iteration (if we run many steps) repeat the search once.
        if (c_search_round >= 10) and cur_round == (c_search_round - 1):
            c = upper_bound
        # early stop: record previous result to avoid stuck
        prev = float('inf')

        for cur in range(attack_iterations):
            # generate the adversarial example
            adv_input = (torch.tanh(t_image + perturbation_layer) + 1) / 2
            # calculate the distance between AE and original image, first part of the loss function
            l2_squared = (adv_input - images).flatten(1).square().sum(1)
            l2 = l2_squared.detach().sqrt()
            # get model output on AE
            output = model(adv_input)

            # in first run, set up the target variable for the loss function
            print(t_0)
            print(output)
            if cur_round == 0 and cur == 0:
                label_onehot = torch.zeros_like(output).scatter(1, t_0.unsqueeze(1), 1)
                label_infhot = torch.zeros_like(output).scatter(1, t_0.unsqueeze(1), float('inf'))
            # get prediction of the model
            prediction = output.argmax(1)
            # check if current image is an adversarial example
            is_adv = prediction != t_0
            # record the most similar AE
            is_similar = l2 < best_l2
            # compare current AE with the global best
            global_is_similar = l2 < global_best_l2
            local_adv_found = is_adv & is_similar
            global_best_adv_found = is_adv & global_is_similar
            # update the local best
            if local_adv_found:
                best_l2 = l2
            # update the global best and save the best AE
            if global_best_adv_found:
                global_best_l2 = l2
                global_best_adv = adv_input.detach()

            # calculate the loss
            partial_loss = get_partial_loss(output, t_0, label_infhot)
            loss = get_loss(l2_squared, c * (partial_loss + transfer_param).clamp_(min=0))

            # early stop
            if i % (attack_iterations // 10) == 0:
                if (loss > prev * 0.9999).all():
                    break
                prev = loss.detach()

            optimizer.zero_grad(set_to_none=True)
            perturbation_layer.grad = grad(loss.sum(), perturbation_layer, only_inputs=True)[0]
            optimizer.step()

        # update c
        if local_adv_found:
            upper_bound = torch.min(upper_bound, c)
        else:
            lower_bound = torch.max(lower_bound, c)

        is_similar = upper_bound < 1e9
        # if can find adv, update the c to a smaller value
        if is_similar:
            c = lower_bound + upper_bound / 2
        elif not local_adv_found:
            # in the case no AE can be found, make c greater
            c *= 10

    # return the best adv
    return global_best_adv


# test the performance of attack
testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)


def get_target(labels):
    a = random.randint(0, 9)
    while a == labels[0]:
        a = random.randint(0, 9)
    return torch.tensor([a])


total = 0
success = 0
num_image = 10  # number of images to be attacked
adv_image_list = []

for i, (images, labels) in enumerate(testloader):
    target_label = get_target(labels)
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = outputs.max(1)
    if predicted.item() != labels.item():
        continue

    total += 1

    # convert an image into adversarial example
    adv_image = zoo_attack(network=model, images=images, t_0=labels)
    adv_image = adv_image.to(device)
    adv_image_list.append(adv_image)
    # m
    adv_output = model(adv_image)
    _, adv_pred = adv_output.max(1)
    if adv_pred.item() != labels.item():
        success += 1

    if total >= num_image:
        break

print('success rate : %.4f' % (success / total))


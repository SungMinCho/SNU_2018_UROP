import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from tqdm import tqdm

NCOLS = 70

def evaluate(model, testloader, loss_fn):
  loss_sum = 0
  correct_sum = 0
  testlen = len(testloader.dataset)
  for x,y in tqdm(testloader, ncols=NCOLS):
    output = model(x)
    _, pred = torch.max(output, 1)
    loss = loss_fn(output, y)

    loss_sum += loss.item()
    correct_sum += torch.sum(pred == y)
  loss_avg = loss_sum / testlen
  acc = correct_sum.double() / testlen
  print("acc : {:.4f} loss : {:.4f}".format(acc, loss_avg))

def train(model, trainloader, optimizer, loss_fn, valloader=None, epochs=10, save_path=None):
  result = {}
  result['acc'] = []
  result['loss'] = []
  validate = True if valloader is not None else False
  save = True if save_path is not None else False
  trainlen = len(trainloader.dataset)
  if validate:
    vallen = len(valloader.dataset)
    result['val_acc'] = []
    result['val_loss'] = []
  if save:
    state_dict = model.state_dict()
    best_acc = 0.0

  for epoch in range(epochs):
    print("Epoch : {}".format(epoch+1))
    loss_sum = 0
    correct_sum = 0
    val_loss_sum = 0
    val_correct_sum = 0

    # train
    for x,y in tqdm(trainloader, ncols=NCOLS):
      optimizer.zero_grad()
      output = model(x)
      _, pred = torch.max(output, 1)
      loss = loss_fn(output, y)
      loss.backward()
      optimizer.step()

      loss_sum += loss.item() * x.size(0)
      correct_sum += torch.sum(pred == y)

    # validate
    if validate:
      for x,y in tqdm(valloader, ncols=NCOLS):
        output = model(x)
        _, pred = torch.max(output, 1)
        loss = loss_fn(output, y)

        val_loss_sum += loss.item() * x.size(0)
        val_correct_sum += torch.sum(pred == y)

    # report
    loss_epoch = loss_sum / trainlen
    acc_epoch = correct_sum.double() / trainlen
    result['loss'].append(loss_epoch)
    result['acc'].append(acc_epoch)
    current_acc = acc_epoch
    print("Loss : {:.4f} Acc : {:.4f}".format(loss_epoch, acc_epoch))
    if validate:
      val_loss_epoch = val_loss_sum / vallen
      val_acc_epoch= val_correct_sum.double()  / vallen
      result['val_loss'].append(val_loss_epoch)
      result['val_acc'].append(val_acc_epoch)
      current_acc = val_acc_epoch
      print("val Loss : {:.4f} val Acc : {:.4f}".format(val_loss_epoch, val_acc_epoch))
    if save and current_acc > best_acc:
      best_acc = current_acc
      state_dict = model.state_dict()

  if save:
    torch.save(state_dict, save_path)

  return result

def distill(teacher, student, trainloader, optimizer, loss_fn, valloader=None, epochs=10,
            save_path=None, temperature=5.0, lambda_const=0.93):
  def distill_loss(output, y, x):
    T = temperature
    true_loss = nn.CrossEntropyLoss()(output, y)

    teacher_y = teacher(x)
    teacher_y_soft = teacher_y / temperature
    output_soft = output / temperature

    return nn.KLDivLoss()(F.log_softmax(output / T), nn.functional.softmax(teacher_y / T)) * (T*T * 2.0 * lambda_const) + F.cross_entropy(output, y) * (1. - lambda_const)

  result = {}
  result['acc'] = []
  result['loss'] = []
  result['true_loss'] = []

  validate = True if valloader is not None else False
  save = True if save_path is not None else False
  trainlen = len(trainloader.dataset)
  if validate:
    vallen = len(valloader.dataset)
    result['val_acc'] = []
    result['val_loss'] = []
    result['val_true_loss'] = []
  if save:
    state_dict = student.state_dict()
    best_acc = 0.0

  for epoch in range(epochs):
    print("Epoch : {}".format(epoch+1))
    loss_sum = 0
    true_loss_sum = 0
    correct_sum = 0
    val_loss_sum = 0
    val_true_loss_sum = 0
    val_correct_sum = 0

    # train
    for x,y in tqdm(trainloader, ncols=NCOLS):
      optimizer.zero_grad()
      output = student(x)
      _, pred = torch.max(output, 1)
      loss = distill_loss(output, y, x)
      loss.backward()
      optimizer.step()

      loss_sum += loss.item() * x.size(0)
      correct_sum += torch.sum(pred == y)

      true_loss = loss_fn(output, y)
      true_loss_sum += true_loss.item() * x.size(0)

    # validate
    if validate:
      for x,y in tqdm(valloader, ncols=NCOLS):
        output = student(x)
        _, pred = torch.max(output, 1)
        loss = distill_loss(output, y, x)

        val_loss_sum += loss.item() * x.size(0)
        val_correct_sum += torch.sum(pred == y)

        true_loss = loss_fn(output, y)
        val_true_loss_sum += true_loss.item() * x.size(0)

    # report
    loss_epoch = loss_sum / trainlen
    true_loss_epoch = true_loss_sum / trainlen
    acc_epoch = correct_sum.double() / trainlen
    result['loss'].append(loss_epoch)
    result['acc'].append(acc_epoch)
    result['true_loss'].append(true_loss_epoch)
    current_acc = acc_epoch
    print("Loss : {:.4f} Acc : {:.4f} True Loss : {:.4f}".format(loss_epoch, acc_epoch, true_loss_epoch))
    if validate:
      val_loss_epoch = val_loss_sum / vallen
      val_true_loss_epoch = val_true_loss_sum / vallen
      val_acc_epoch= val_correct_sum.double()  / vallen
      result['val_loss'].append(val_loss_epoch)
      result['val_true_loss'].append(val_true_loss_epoch)
      result['val_acc'].append(val_acc_epoch)
      current_acc = val_acc_epoch
      print("val Loss : {:.4f} val Acc : {:.4f} val True Loss : {:.4f}".format(val_loss_epoch, val_acc_epoch, val_true_loss_epoch))
    if save and current_acc > best_acc:
      best_acc = current_acc
      state_dict = student.state_dict()

  if save:
    torch.save(state_dict, save_path)

  return result

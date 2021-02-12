#from typing import ForwardRef
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard
from torch.utils.tensorboard.writer import SummaryWriter


def starting_train(
    train_dataset, val_dataset, model, hyperparameters, n_eval, summary_path
):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
        summary_path:    Path where Tensorboard summaries are located.
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    #initialize summary logs
    train_summary = SummaryWriter("./logs/CassavaLogs")
    validation_summary = SummaryWriter("./logs/CassavaLogs")

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    # Initialize summary writer (for logging)
    writer = torch.utils.tensorboard.SummaryWriter(summary_path)

    step = 0
    n_eval_val = 1000
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for i, batch in enumerate(train_loader):
            print(f"\rIteration {i + 1} of {len(train_loader)} ...", end="")

            # TODO: Backpropagation and gradient descent
   
            # Input data for this batch
            input_data, labels = batch

            # Zero/clear out the gradients
            optimizer.zero_grad()

            # Make predictions and calculate loss
            predictions = model.forward(input_data)
            loss = loss_fn(predictions, labels)

            # backprop advance 
            loss.backward()
            optimizer.step()

            print("Epoch ", epoch, "\n  Train Loss: ", loss.item())
            
            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0:
                #predictions = ForwardRef
                # TODO:
                # Compute training loss and accuracy.
                ####
                # Log the results to Tensorboard.
                train_summary.add_scalar("train_loss", loss, global_step = step)
                train_acc = compute_accuracy(predictions, labels)
                train_summary.add_scalar("train_acc", train_acc, global_step = step)
                print('====== TRAIN ACC:', train_acc)


                # TODO:
                # Compute validation loss and accuracy.
                '''
                evaluate(val_loader, model, loss_fn, validation_summary, val_dataset, loss, step,batch_size)
                '''
                model.train()

            # Evaluate on validation dataset
            if step % n_eval_val == 0:
                evaluate(val_loader, model, loss_fn, validation_summary, val_dataset, loss, step,batch_size)
                model.train()

            step += 1


def compute_accuracy(outputs, labels):
    #print(outputs)
    #print(torch.topk(outputs,1)[1])
    #print(list(labels.size()))
    #print(labels)
    n_correct = (torch.transpose(torch.topk(outputs,1)[1],0,1) == labels).sum().item()
    n_total = len(outputs)
    #print('   -- Acc:', n_correct / n_total)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn, validation_summary, val_dataset, loss, step,batch_size):
    
    """
    Computes the loss and accuracy of a model on the validation dataset.
    """
    model.eval()
    validation_N = 1024

    #gets first validation_VAL images for validation testing
    #for i in validation_N:
    #    input_data_VAL
    loss_VAL = 0 

    acc_sum = 0
    acc_times = 0
    for i, batch in enumerate(val_loader):
        input_data_VAL, labels_VAL = batch
        if i >= validation_N/batch_size:
            break
        
        predictions_VAL = model.forward(input_data_VAL)
        acc_sum = acc_sum + compute_accuracy(predictions_VAL, labels_VAL)
        acc_times = acc_times + 1
    
    validation_summary.add_scalar("val_acc", acc_sum/acc_times, global_step = step)
    print('===== VALIDATION ACC:', acc_sum/acc_times,' =====')
        

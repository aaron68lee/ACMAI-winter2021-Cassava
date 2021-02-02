from typing import ForwardRef
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
            """
            print("=========================================================")
            print(labels.shape)
            #print(labels)
            print(predictions.shape)
            #print(predictions)
            print("=========================================================")
            """
            loss = loss_fn(predictions, labels)

            # backprop advance 
            loss.backward()
            optimizer.step()

            print("Epoch \n", epoch, "  Train Loss: ", loss.item())
            """
            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0:
                #predictions = ForwardRef
                # TODO:
                # Compute training loss and accuracy.
                ####
                # Log the results to Tensorboard.
                train_summary.add_scalar("train_loss", loss, global_step = step)

                # TODO:
                # Compute validation loss and accuracy.
                evaluate(val_loader, model, loss_fn, validation_summary, val_dataset, loss, step)
                model.train()
            """

            step += 1

    #print(compute_accuracy(outputs, labels))

def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """

    n_correct = (torch.round(outputs) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn, validation_summary, val_dataset, loss, step):
    
    """
    Computes the loss and accuracy of a model on the validation dataset.
    """
    model.eval()

    input_data_VAL, labels_VAL = val_dataset

    predictions_VAL = model.forward(input_data_VAL)
    loss_VAL = loss_fn(predictions_VAL, labels_VAL)

    # Log the results to Tensorboard.
    validation_summary.add_scalar("validation_loss", loss_VAL, global_step = step)

    print("  Validation Loss: ", loss.item())

    # Don't forget to turn off gradient calculations!
    
    pass

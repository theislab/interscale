from graph_transformer_long_range_niches.tl.evaluation import accuracy  # noqa, register custom modules

import numpy as np
import wandb
import torch

def train_gnntransformer(train_loader, model, optimizer, criterion, wandb_use, N_EPOCHS=20):

    # Data for animations
    losses = []
    accuracies_transformer = []
    accuracies_gnn = []
    outputs = []

    # Training loop
    for epoch in range(N_EPOCHS):
        curr_acc_trans = []
        curr_acc_gnn = []
        curr_loss = []
        for idx, batch in enumerate(train_loader): 
            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            out_gnn, out_transformer, index_nodes = model(batch) # [B, C] with C being the number of tasks to predict, e.i. 
                
            # Calculate loss function
            y_true = []
            for i in range(batch.batch[-1] + 1):
                mask = batch.batch.eq(i)
                y_true += batch.y[mask][index_nodes[i]]
            loss = criterion(out_transformer, torch.LongTensor(y_true))
            # Calculate accuracy
            #print('Predicted label: ', out.argmax(dim=1), 'True label', batch.y)
            acc_transformer = accuracy(out_transformer.argmax(dim=1), torch.LongTensor(y_true))
            acc_gnn = accuracy(out_gnn.argmax(dim=1), batch.y)

            # Compute gradients
            loss.backward(retain_graph=True)

            # Tune parameters
            optimizer.step()

            # Store data for animations
            #embeddings.append(h)
            curr_loss.append(loss.detach().item())
            curr_acc_trans.append(acc_transformer)
            curr_acc_gnn.append(acc_gnn)
            outputs.append(out_transformer.argmax(dim=1))
            

        # Print metrics every 10 epochs
        if epoch % 5 == 0:
            print(f'Transformer Epoch {epoch:>3} | Loss: {np.mean(curr_loss):.2f} | GNN acc: {np.mean(curr_acc_gnn)*100:.2f}% | Trans acc: {np.mean(curr_acc_trans)*100:.2f}%')
            accuracies_transformer.append(np.mean(curr_acc_trans))
            accuracies_gnn.append(np.mean(curr_acc_gnn))
            losses.append(np.mean(curr_loss))

    return accuracies_transformer, accuracies_gnn


def train_gnn(train_loader, model, optimizer, criterion, wandb_use, N_EPOCHS=20):

    # Data for animations
    losses = []
    accuracies_gnn = []
    outputs = []

    # Training loop
    for epoch in range(N_EPOCHS):
        curr_acc_gnn = []
        curr_loss = []
        for idx, batch in enumerate(train_loader): 
            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            gnn_x, gnn_z = model(batch.x, batch.edge_index) # [B, C] with C being the number of tasks to predict, e.i. 

            # In case of multiple graphs in one batch -> merge the index lists
            # for i in range(batch.batch[-1]+1):
                
            # Calculate loss function
            loss = criterion(gnn_z, batch.y)

            # Calculate accuracy
            #print('Predicted label: ', out.argmax(dim=1), 'True label', batch.y)
            acc_gnn = accuracy(gnn_z.argmax(dim=1), batch.y)

            # Compute gradients
            loss.backward(retain_graph=True)

            # Tune parameters
            optimizer.step()

            # Store data for animations
            #embeddings.append(h)
            curr_loss.append(loss.detach().item())
            curr_acc_gnn.append(acc_gnn)
            outputs.append(gnn_z.argmax(dim=1))
        
        if wandb_use:
            wandb.log({"epoch": epoch,
                        "loss": np.mean(curr_loss),
                        "acc GNN": np.mean(curr_acc_gnn)})

        # Print metrics every 10 epochs
        if epoch % 5 == 0:
            print(f'Transformer Epoch {epoch:>3} | Loss: {np.mean(curr_loss):.2f} | GNN acc: {np.mean(curr_acc_gnn)*100:.2f}% ')
            accuracies_gnn.append(np.mean(curr_acc_gnn))
            losses.append(np.mean(curr_loss))
            

    return accuracies_gnn
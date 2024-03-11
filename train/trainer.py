import torch
import time

def train_loop(args, dataloader, model, optimizer, loss_fn, model_save_path):
    model.train()

    loss_max = 100
    
    total_loss = []
    size = len(dataloader)

    for epoch in range(args.epochs):
        print(f"Start epoch : {epoch+1}/{args.epochs}!")
        epoch_loss = 0
        epoch_start = time.time()

        for batch, (ct, hippo, edge) in enumerate(dataloader):
            ct = ct.to(args.device).float()
            hippo = hippo.to(args.device).float()
            edge = edge.to(args.device).float()

            output, output_edge= model(ct)

            loss1_1, loss1_2 = loss_fn(hippo, output)
            loss2_1, loss2_2 = loss_fn(edge, output_edge)

            loss1 = loss1_1 + loss1_2 
            loss2 = loss2_1 + loss2_2 

            loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss

            if batch % 5 == 0:
                print(f"Batch loss => {loss:>.5f} = {loss1_1:>.5f} + {loss1_2:>.5f} + {loss2_1:>.5f} + {loss2_2:>.5f} {batch}/{size}")

        # if loss < loss_max:
        #     torch.save(model.state_dict(), model_save_path)
        #     loss_max = loss
        #     print(f"Model Saved!")

        epoch_end = time.time()
        print(f"    Epoch Loss : {epoch_loss/size:>.5f}")
        print(f"    Epoch Time : {(epoch_end - epoch_start) // 60} min {(epoch_end  - epoch_start) % 60} sec")
        print(f"End Epoch : {epoch+1}/{args.epochs}")
        print()

        total_loss.append(epoch_loss.detach().cpu()/size)

    torch.save(model.state_dict(), model_save_path)

    return total_loss
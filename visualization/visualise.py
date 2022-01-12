
import matplotlib.pyplot as plt


def visualise(model, dl_test, N):
    
    iterator = iter(dl_test)
    
    fig, axes = plt.subplots(N,2,figsize=(12,N*6))
    for i in range(N):
        data, mask, image = next(iterator)

        axes[i, 0].set_title("Ground truth")
        axes[i, 0].imshow(image.squeeze())
        axes[i, 0].imshow(mask.squeeze(), alpha=0.4)

        logits = model(data)
        preds = logits.detach().cpu().numpy().argmax(axis=1).squeeze()

        axes[i, 1].set_title("Prediction")
        axes[i, 1].imshow(image.squeeze())
        axes[i, 1].imshow(preds.squeeze(), alpha=0.4)
        
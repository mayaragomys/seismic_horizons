import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def calculate_IoU(gt, y_pred):
    intersection = y_pred * gt
    union = y_pred + gt - intersection
    jacard = np.sum(intersection) / np.sum(union)

    return jacard

def calculate_dice(gt, y_pred):
    intersection = y_pred * gt
    dice = (2.0 * np.sum(intersection)) / (np.sum(y_pred) + np.sum(gt))

    return dice

def compute_metrics(true_labels, predicted_labels):
    accuracy = 100*accuracy_score(true_labels, predicted_labels)
    f1score = 100*f1_score(true_labels, predicted_labels, average=None)
    recall = 100*recall_score(true_labels, predicted_labels, average=None)
    precision = 100*precision_score(true_labels, predicted_labels, average=None)
    return accuracy, f1score, recall, precision

def calculate_metrics(imgs, preds):
    dice_total = 0
    jaccard_total = 0
    # arredondando os valores dos predicts
    preds = np.round(preds, 0)
    print(np.array(preds).shape, np.array(imgs).shape)

    for idx, img in enumerate(imgs):
        gt = np.reshape(img, (img.shape[0], img.shape[1]))
        gt = np.array(gt, dtype=np.float32)
        y_pred = np.reshape(preds[idx], (img.shape[0], img.shape[1]))
    
        #calcula métricas
        dice = calculate_dice(gt, y_pred)
        jaccard = calculate_IoU(gt, y_pred)
        dice_total += dice
        jaccard_total += jaccard
    
    imgs = np.array(imgs)
    preds = np.array(preds)
    print("Dice: ", dice_total/len(imgs))
    print("Jaccard: ", jaccard_total/len(imgs))
    accuracy, f1score, recall, precision = compute_metrics(imgs.flatten(), preds.flatten())
    metrics = {}
    metrics["Dice:"] = dice_total/len(imgs)
    metrics["Jaccard:"] = jaccard_total/len(imgs)
    metrics["accuracy:"] = accuracy
    metrics["f1score:"] = f1score
    metrics["recall:"] = recall
    metrics["precision:"] = precision
    
    return metrics

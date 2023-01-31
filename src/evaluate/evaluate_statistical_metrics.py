import numpy as np
import torch
from scipy import linalg

def evaluate_fid(gt_joint, out_jt, num_labels, classifier, gt_labels):
    with torch.no_grad():
        fid = np.zeros(len(gt_joint))
        for idx in range(len(gt_joint)):
            y_pred_gt, ground_truth_activations = classifier.predict(gt_joint[idx])
            statistics_1 = calculate_activation_statistics(ground_truth_activations)
            y_pred, pred_activations = classifier.predict(out_jt[idx])
            statistics_2 = calculate_activation_statistics(pred_activations)
            fid[idx] = calculate_frechet_distance(statistics_1[0], statistics_1[1],
                                      statistics_2[0], statistics_2[1])
        final_fid = np.mean(fid)
        return final_fid


def calculate_activation_statistics(activations):
    activations = activations.cpu().numpy()
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma

def calculate_accuracy(joints_3d_vec, num_labels, classifier, gt_labels):
    confusion = torch.zeros(num_labels, num_labels, dtype=torch.long)
    with torch.no_grad():
        for idx in range(len(joints_3d_vec)):
            y_pred, _ = classifier.predict(joints_3d_vec[idx])
            batch_pred = y_pred.max(dim=1).indices
            batch_label = gt_labels[idx].max(dim=1).indices
            for label, pred in zip(batch_label, batch_pred):
                # print(label.data, pred.data)
                confusion[label][pred] += 1
    return np.trace(confusion.numpy())/len(joints_3d_vec)

def calculate_diversity_(activations, labels, num_labels):
    diversity_times = 200
    
    num_motions = len(labels)
    diversity = 0
    first_indices = np.random.randint(0, num_motions, diversity_times)
    second_indices = np.random.randint(0, num_motions, diversity_times)
    for first_idx, second_idx in zip(first_indices, second_indices):
        diversity += torch.dist(activations[first_idx, :],
                                activations[second_idx, :])
    diversity /= diversity_times
    return diversity

def calculate_multimodality_(activations, labels, num_labels):
    num_motions = len(labels)
    multimodality = 0
    multimodality_times = 20
    labal_quotas = np.repeat(multimodality_times, num_labels)
    count = 0
    while np.any(labal_quotas > 0) and count <= 10000:
        count+=1
        first_idx = np.random.randint(0, num_motions)
        first_label = labels[first_idx].max(dim=0).indices
        if not labal_quotas[first_label.cpu().detach().numpy()]:
            continue
        second_idx = np.random.randint(0, num_motions)
        second_label = labels[second_idx].max(dim=0).indices
        while first_label != second_label:
            second_idx = np.random.randint(0, num_motions)
            second_label = labels[second_idx].max(dim=0).indices
        labal_quotas[first_label] -= 1
        first_activation = activations[first_idx, :]
        second_activation = activations[second_idx, :]
        multimodality += torch.dist(first_activation,
                                    second_activation)
    multimodality /= (multimodality_times * num_labels)
    return multimodality

def calc_AVE(joints_sbj, joints_gt):
    T = joints_gt.shape[0]    
    J = joints_gt.shape[1]
    var_gt = torch.zeros((J))
    var_pred = torch.zeros((J))
    for j in range(J):
        var_gt[j] = (torch.sum(joints_gt[:, j]) - torch.mean(joints_gt[:, j]))**2/(T-1)
        var_pred[j] = (torch.sum(joints_sbj[:, j]) - torch.mean(joints_sbj[:, j]))**2/(T-1)
    mean_ave_loss = mean_l2di_(var_gt, var_pred)
    return mean_ave_loss

def mean_l2di_(a,b):
    x = torch.mean(torch.sqrt(torch.sum((a - b)**2, -1)))
    return x

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'
    diff = mu1 - mu2
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)
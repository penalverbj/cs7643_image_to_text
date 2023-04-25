import torch.nn as nn


def cosine_similarity_loss(prediction_embeddings, target_embeddings):
    """_summary_

    :param pred: _description_
    :type pred: _type_
    :param target: _description_
    :type target: _type_
    :return: _description_
    :rtype: _type_
    """
    cos = nn.CosineSimilarity(dim=1)
    output = -cos(prediction_embeddings, target_embeddings).mean()
    return output

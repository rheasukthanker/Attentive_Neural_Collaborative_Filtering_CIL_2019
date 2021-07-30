import abc
import numpy as np


class AbstractSampleWeighting():
    def __init__(self, full_data):
        raise NotImplementedError("overwritte")

    def get_batch_weights(self, user_ids, item_ids, ratings):
        raise NotImplementedError("overwritte")



class Weighted_Loss_Ratings:
    def __init__(self, full_data):
        self.ratings_weight_dict = self.get_ratings_distribution(full_data)

    def get_ratings_distribution(self, full_data):
        rating_dict = {}
        count1 = full_data[full_data.loc[:, "prediction"] == 1].shape[0]
        count2 = full_data[full_data.loc[:, "prediction"] == 2].shape[0]
        count3 = full_data[full_data.loc[:, "prediction"] == 3].shape[0]
        count4 = full_data[full_data.loc[:, "prediction"] == 4].shape[0]
        count5 = full_data[full_data.loc[:, "prediction"] == 5].shape[0]
        counts = np.array([count1, count2, count3, count4, count5])
        sum_counts = np.sum(counts)
        rating_dict[1] = 1 / (count1 / sum_counts)
        rating_dict[2] = 1 / (count2 / sum_counts)
        rating_dict[3] = 1 / (count3 / sum_counts)
        rating_dict[4] = 1 / (count4 / sum_counts)
        rating_dict[5] = 1 / (count5 / sum_counts)
        return rating_dict

    def get_batch_weights(self, user_ids, item_ids, ratings):
        #def get_weights_for_labels(labels,weights_dict):
        arr = []
        for x in ratings:
            x = int(x)
            arr.append(self.ratings_weight_dict[x])
        return np.array(arr)

class Weighted_Loss_Users:
    def __init__(self, full_data):
        self.users_weight_dict = self.get_users_distribution(full_data)

    def get_users_distribution(self, full_data):
        user_dict = {}
        sum_all_known = full_data.shape[0]
        print(sum_all_known)
        for i in range(0, 10000):
            user_known_count = full_data[full_data.loc[:, "row_id"] == i].shape[0]
            user_dict[i] = 1 / (user_known_count / sum_all_known)
        factor=1.0/sum(user_dict.values())
        normalized_user_dict = {k: v*factor for k, v in user_dict.items() }
        return normalized_user_dict

    def get_batch_weights(self, user_ids, item_ids, ratings):
        # def get_weights_for_labels(labels,weights_dict):
        arr = []
        # print(self.users_weight_dict)
        for x in user_ids:
            x = int(x)
            arr.append(self.users_weight_dict[x])
        return np.array(arr)

class Weighted_Debug(AbstractSampleWeighting):
    def __init__(self, _):
        pass
    def get_batch_weights(self, user_ids, item_ids, ratings):
        print('get_weights')
        return np.zeros_like(ratings)

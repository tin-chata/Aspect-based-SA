"""
Created on Tue Mar  6 15:12:42 2018

@author: dtvo
"""
import numpy as np
import networkx as nx


def emb_term(input_term, W, vocab):
    for idx, term in enumerate(input_term.split(' ')):
        if term in vocab:
            # print('Word: %s  Position in vocabulary: %i' % (term, vocab[term]))
            if idx == 0:
                vec_result = np.copy(W[vocab[term]])
            else:
                vec_result += W[vocab[term]]
        else:
            # print('Word: %s  Out of dictionary!\n' % term)
            return np.zeros(1)
    vec_norm = np.zeros(vec_result.shape)
    d = (np.sum(vec_result ** 2, ) ** (0.5))
    vec_norm = (vec_result.T / d).T
    return vec_norm


def norm_embs(emb_file):
    with open(emb_file, 'r') as f:
        vectors = {}
        words = []
        for line in f:
            vals = line.rstrip().split(' ')
            if len(vals) == 2:
                continue
            vectors[vals[0]] = [float(x) for x in vals[1:]]
            words.append(vals[0])
    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word]] = v

    # normalize each word vector to unit variance
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T
    return (W_norm, vocab, ivocab)


W, vocab, ivocab = norm_embs("/Users/duytinvo/Projects/aspectSA/data/w2v/sf_hotel.pro.v2.vec")
# group_terms = {
#     "location": ["location"],                               # Location
#     "price": ["price"],                                     # Price
#     "staff": ["staff"],                                     # Staff
#     "ambiance": ["ambiance"],                               # Ambiance
#     "service": ["services"],                                # Service and facility
#     "transportation": ["transportation", "parking"],        # transportation and parking
#     "food": ["food"],                                       # food and beverage
#     "room": ["room", "electricity"]                         # in-room facility
# }

group_terms = {
    "room": ["room", "rooms"],  # room
    "price": ["price", "bill", "cost", "discount",
              "expenditure", "fare", "fee", "payment",
              "rate", "tariffs"],  # value
    "service": ["staff", "support", "service"],  # Staff
    "ambiance": ["ambiance"],  # Ambiance
    "food": ["food"],  # food and beverage
    "location": ["location", "area", "district", "locale", "neighborhood", "region", "spot", "venue"],  # Location
    "amenity": ["amenities", "facilities"],  # Service and facility
}


def Cosinesim(vec1, vec2):
    return np.dot(vec1 / sum(vec1 ** 2) ** 0.5, vec2 / sum(vec2 ** 2) ** 0.5)


def query_graph(G, aspect):
    aspect_emb = emb_term(aspect, W, vocab)
    if sum(aspect_emb) == 0:
        return (-1.0, -1.0)
    else:
        if G.has_node(aspect):
            return (G.nodes[aspect]["scale"], 10 * G.nodes[aspect]["scale"] * Cosinesim(G.graph["norm_wv"], aspect_emb))
        else:
            return (0.0, Cosinesim(G.graph["norm_wv"], aspect_emb))


# group_graphs = {
#     "location": nx.read_gpickle("/Users/duytinvo/Projects/aspectSA/data/wv_groups/hotel_location_group.gpickle"),                       # Location
#     "price": nx.read_gpickle("/Users/duytinvo/Projects/aspectSA/data/wv_groups/hotel_price_group.gpickle"),                             # Price
#     "staff": nx.read_gpickle("/Users/duytinvo/Projects/aspectSA/data/wv_groups/hotel_staff_group.gpickle"),                             # Staff
#     "ambiance": nx.read_gpickle("/Users/duytinvo/Projects/aspectSA/data/wv_groups/hotel_ambiance_group.gpickle"),                       # Ambiance
#     "Service_&_facility": nx.read_gpickle("/Users/duytinvo/Projects/aspectSA/data/wv_groups/hotel_services_group.gpickle"),                        # Service and facility
#     "transportation_&_parking": nx.read_gpickle("/Users/duytinvo/Projects/aspectSA/data/wv_groups/hotel_transportation_parking_group.gpickle"),   # transportation and parking
#     "food_&_beverage": nx.read_gpickle("/Users/duytinvo/Projects/aspectSA/data/wv_groups/hotel_food_group.gpickle"),                               # food and beverage
#     "in-room_facility": nx.read_gpickle("/Users/duytinvo/Projects/aspectSA/data/wv_groups/hotel_room_electricity_group.gpickle")                    # in-room facility
# }

group_graphs = {
    "room": nx.read_gpickle("/Users/duytinvo/Projects/aspectSA/data/wv_groups/hotel_room_rooms_group.gpickle"),  # room
    "price": nx.read_gpickle("/Users/duytinvo/Projects/aspectSA/data/wv_groups/hotel_price_bill_cost_discount_expenditure_fare_fee_payment_rate_tariffs_group.gpickle"),  # value
    "service": nx.read_gpickle("/Users/duytinvo/Projects/aspectSA/data/wv_groups/hotel_staff_support_service_group.gpickle"),  # Staff
    "ambiance": nx.read_gpickle("/Users/duytinvo/Projects/aspectSA/data/wv_groups/hotel_ambiance_group.gpickle"),  # Ambiance
    "food": nx.read_gpickle("/Users/duytinvo/Projects/aspectSA/data/wv_groups/hotel_food_group.gpickle"),  # food and beverage
    "location": nx.read_gpickle("/Users/duytinvo/Projects/aspectSA/data/wv_groups/hotel_location_area_district_locale_neighborhood_region_spot_venue_group.gpickle"),  # Location
    "amenity": nx.read_gpickle("/Users/duytinvo/Projects/aspectSA/data/wv_groups/hotel_amenities_facilities_group.gpickle"),  # Service and facility
}


def grouping(np):
    distance = []
    for g in group_graphs:
        d = query_graph(group_graphs[g], np)
        distance.append((g, d))
    return sorted(distance, key=lambda x: x[1][-1], reverse=True)


if __name__ == "__main__":
    N = 20          # number of closest words that will be shown
    distance = grouping("breakfast")
    # build_graphs()





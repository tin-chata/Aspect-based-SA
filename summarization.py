"""
Created on 2018-09-06
@author: duytinvo
"""
import os
import csv
from collections import defaultdict, Counter

review_file = "/Users/duytinvo/Projects/aspectSA/data/customer_reviews/Hotel_Review-g181808-d7332235-Reviews-Hampton_Inn_Suites_Airdrie-Airdrie_Alberta.html.csv"
pdir, bname = os.path.split(review_file)
bdir = bname.split(".")[0]
pathdir = os.path.join(pdir, bdir)


groups = {
    "location": "location".upper() + "_group.csv",                       # Location
    "price": "price".upper() + "_group.csv",                             # Price
    "staff": "staff".upper() + "_group.csv",                             # Staff
    "ambiance": "ambiance".upper() + "_group.csv",                       # Ambiance
    "Service_&_facility": "Service_&_facility".upper() + "_group.csv",                        # Service and facility
    "transportation_&_parking": "transportation_&_parking".upper() + "_group.csv",   # transportation and parking
    "food_&_beverage": "food_&_beverage".upper() + "_group.csv",                               # food and beverage
    "in-room_facility": "in-room_facility".upper() + "_group.csv"                    # in-room facility
}

sum_file = os.path.join(pathdir, "summary.txt")
with open(sum_file, "w") as g:
    for k in groups:
        filename = os.path.join(pathdir, groups[k])
        aspects = defaultdict(list)
        count_aspects = Counter()
        pol_g = 0.
        with open(filename, "r") as f:
            csvreader = csv.reader(f)
            for line in csvreader:
                date, aspect, porlarity_score, sentence, review_score = line
                aspect = aspect.lower()
                aspects[aspect].append((float(porlarity_score), sentence))
                count_aspects.update([aspect])
                pol_g += float(porlarity_score)

        pol_g = pol_g/sum(count_aspects.values())
        g.write("-" * 200 + "\n")
        g.write("---" + "\t\t\t\t\t\t\t Group: %s\n" % k.upper())
        g.write("---" + "\t\t\t\t\t\t Average polarity score: %f\n"%pol_g)
        g.write("-" * 200 + "\n\n")
        statistics = []
        for ap, num in count_aspects.most_common(10):
            sorted_ap = sorted(aspects[ap], reverse=True)
            scores, sents = zip(*sorted_ap)
            mean = sum(scores)/len(scores)
            g.write("\t" + "~" * 150 + "\n")
            g.write("\t+ Common aspect: %s (%d times)\n" % (ap,num))
            g.write("\t+ Average polarity score: %f\n" % mean)
            if num < 10:
                best5 = sents[:len(sents) // 2]
                worst5 = sents[-len(sents) // 2:]
            else:
                best5 = sents[:5]
                worst5 = sents[-5:]
            g.write("\t+ Top 5 best: \n")
            for rv in best5:
                g.write("\t\t* %s\n"%rv)
            g.write("\t+ Top 5 worst: \n")
            for rv in worst5:
                g.write("\t\t* %s\n" % rv)
            g.write("\n")
            statistics.append((ap,mean, best5, worst5))
        # g.write("="*200 + "\n")
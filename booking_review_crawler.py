"""
Created on 2018-09-19
@author: duytinvo
"""
import time
import requests
from bs4 import BeautifulSoup
from sklearn.externals import joblib


def review_url(hotel_link):
    result = requests.get(hotel_link)
    content = result.content
    soup = BeautifulSoup(content, "html.parser")
    rvbtns = soup.find_all("a", attrs={"class": "show_all_reviews_btn"})
    reviews_link = ""
    if len(rvbtns) == 1:
        reviews_link = rvbtns[0].get("href").strip()
    return reviews_link


def review_rp(hotel_link):
    return hotel_link.split("?")[0].replace("/hotel/ca/", "/reviews/ca/hotel/")


def url1page(region_link, booking_link):
    result = requests.get(region_link)
    content = result.content
    soup = BeautifulSoup(content, "html.parser")
    hotel_links = []
    hlinks = soup.find_all("div",
                           attrs={"class": 'sr_item sr_item_new sr_item_default sr_property_block sr_item_no_dates '})
    for hlink in hlinks:
        hname = hlink.find_all("span", attrs={"class": "sr-hotel__name"})[0].get_text().strip()
        hurl = hlink.find_all("a", attrs={"class": "hotel_name_link url"})[0].get("href").strip()
        revs = hlink.find_all("span", attrs={"class": 'review-score-widget__subtext'})
        if len(revs) == 1:
            no_revs = int(revs[0].get_text().strip().split(" ")[0].replace(",", ""))
            if no_revs >= 50:
                rvurl = review_rp(hurl)
                hurl = booking_link + hurl
                rvurl = booking_link + rvurl
                hotel_links.append((hname, no_revs, hurl, rvurl))

    nxtbtns = soup.find_all("a", attrs={"title": "Next page"})
    nxt_link = ""
    if len(nxtbtns) == 1:
        nxt_link = nxtbtns[0].get("href")
    return hotel_links, nxt_link


def hotel_urls(region_link, booking_link, max_page=5):
    c = 0
    list_hotel, nxt_link = url1page(region_link, booking_link)
    while len(nxt_link) != 0:
        print(nxt_link)
        if max_page > 0:
            c += 1
            if c == max_page:
                break
        hotels, nxt_link = url1page(nxt_link, booking_link)
        list_hotel.extend(hotels)
        # TODO: write to a file instead of using list
    list_hotel.sort(key=lambda x: x[1], reverse=True)
    return list_hotel


def write_line(filename, rev):
    with open(filename, "a") as f:
        f.write(rev + "\n")


def rv1page(rv_link, pos_file, neg_file):
    result = requests.get(rv_link)
    content = result.content
    soup = BeautifulSoup(content, "html.parser")
    rvconts = soup.find_all("div", attrs={"class": "review_item_review_content"})
    nxtpage = soup.find_all("a", attrs={"id": "review_next_page_link"})
    nxt_link = ""
    if len(nxtpage) != 0:
        nxt_link = nxtpage[0].get("href").strip()
    pos_rvs = []
    neg_rvs = []
    for rvcont in rvconts:
        pos = rvcont.find_all("p", attrs={"class": "review_pos "})
        if len(pos) == 1:
            pos_rv = pos[0].find_all("span", attrs={"itemprop": "reviewBody"})[0].text.strip()
            pos_rv = " ".join(pos_rv.split())
            if len(pos_rv.split()) >= 2:
                pos_rvs.append(pos_rv)
                # TODO: write into file instead of  using lists
                write_line(pos_file, pos_rv)

        neg = rvcont.find_all("p", attrs={"class": "review_neg "})
        if len(neg) == 1:
            neg_rv = neg[0].find_all("span", attrs={"itemprop": "reviewBody"})[0].text.strip()
            neg_rv = " ".join(neg_rv.split())
            if len(neg_rv.split()) >= 2:
                neg_rvs.append(neg_rv)
                write_line(neg_file, neg_rv)
    return pos_rvs, neg_rvs, nxt_link


def rv1page_pol(rv_link, pos_file, neg_file):
    result = requests.get(rv_link)
    content = result.content
    soup = BeautifulSoup(content, "html.parser")
    rvconts = soup.find_all("div", attrs={"class": "review_item_review"})
    nxtpage = soup.find_all("a", attrs={"id": "review_next_page_link"})
    nxt_link = ""
    if len(nxtpage) != 0:
        nxt_link = nxtpage[0].get("href").strip()
    pos_rvs = []
    neg_rvs = []
    for rvcont in rvconts:
        score = rvcont.find_all("span", attrs={"class": "review-score-badge"})
        if len(score) == 1:
            score = float(score[0].text.strip())
            if score >= 9.0:
                pos = rvcont.find_all("p", attrs={"class": "review_pos "})
                # if len(pos) + len(neg) == 1:
                if len(pos) == 1:
                    pos_rv = pos[0].find_all("span", attrs={"itemprop": "reviewBody"})[0].text.strip()
                    pos_rv = " ".join(pos_rv.split())
                    if len(pos_rv.split()) >= 2:
                        pos_rvs.append(pos_rv)
                        # TODO: write into file instead of  using lists
                        write_line(pos_file, pos_rv)
            if score <= 7.0:
                neg = rvcont.find_all("p", attrs={"class": "review_neg "})
                if len(neg) == 1:
                    neg_rv = neg[0].find_all("span", attrs={"itemprop": "reviewBody"})[0].text.strip()
                    neg_rv = " ".join(neg_rv.split())
                    if len(neg_rv.split()) >= 2:
                        neg_rvs.append(neg_rv)
                        write_line(neg_file, neg_rv)
    return pos_rvs, neg_rvs, nxt_link


def rv1hotel(rv_link, booking_link, pos_file, neg_file, crawler=rv1page):
    start = time.time()
    pos_rvs, neg_rvs, nxt = crawler(rv_link, pos_file, neg_file)
    while len(nxt) != 0:
        nxt = booking_link + nxt
        pos, neg, nxt = crawler(nxt, pos_file, neg_file)
        pos_rvs.extend(pos)
        neg_rvs.extend(neg)
    now = time.time() - start
    print("\t+ Extracted %d negative and %d positive reviews in %.4f(s); speed: %.2f(reviews/s)." % (len(neg_rvs), len(pos_rvs), now, (len(neg_rvs) + len(pos_rvs))/now))
    return pos_rvs, neg_rvs


def rvnhotel(hotel_links, booking_link, pos_file, neg_file, n=5, crawler=rv1page):
    pos_data = []
    neg_data = []
    start = time.time()
    if n >0:
        hotel_links = hotel_links[:n]
    for hotel_link in hotel_links:
        print("- Scraping hotel '%s':" % hotel_link[0])
        rv_link = hotel_link[-1]
        pos_rvs, neg_rvs = rv1hotel(rv_link, booking_link, pos_file, neg_file, crawler)
        pos_data.extend(pos_rvs)
        neg_data.extend(neg_rvs)
    now = time.time() - start
    print("Scraping all %d hotels (%d positive & %d negative) in %.4f(s); average speed %.2f(reviews/s)" %
          (len(hotel_links), len(pos_data), len(neg_data), now, (len(neg_data) + len(pos_data))/now))
    return pos_data, neg_data


if __name__ == "__main__":
    """
    python booking_review_crawler.py --region_file /media/data/hotels/booking_v2/raw_data/booking_britishcolumbia_info.pkl --pos_file /media/data/hotels/booking_v2/raw_data/booking_britishcolumbia_positive.txt --neg_file /media/data/hotels/booking_v2/raw_data/booking_britishcolumbia_negative.txt
    """
    import argparse

    argparser = argparse.ArgumentParser()

    argparser.add_argument('--booking_link', help='Booking link', default="https://www.booking.com", type=str)

    argparser.add_argument('--region_file', help='Regional file',
                           default="/media/data/hotels/booking_v2/raw_data/booking_britishcolumbia_info.pkl",
                           type=str)
    argparser.add_argument('--pos_file', help='Positive file',
                           default="/media/data/hotels/booking_v2/raw_data/booking_britishcolumbia_positive.txt",
                           type=str)
    argparser.add_argument('--neg_file', help='Negative file',
                           default="/media/data/hotels/booking_v2/raw_data/booking_britishcolumbia_negative.txt",
                           type=str)
    args = argparser.parse_args()

    hotel_links = joblib.load(args.region_file)

    pos_data, neg_data = rvnhotel(hotel_links, args.booking_link, args.pos_file, args.neg_file,
                                  n=-1, crawler=rv1page_pol)

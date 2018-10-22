"""
Created on 2018-08-30
@author: duytinvo
"""
import re
import csv
import requests
from datetime import datetime
from bs4 import BeautifulSoup
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def write_csv_a_lines(filename, data):
    with open(filename, "a") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(data)


def region_urls(first_url, region_ID="-g154913-"):
    result = requests.get(first_url)
    content = result.content
    soup = BeautifulSoup(content, "html.parser")
    num_pages = int(soup.find_all("a", attrs={"class":"pageNum last taLnk "})[0].get_text(strip=True))
    list_urls = []
    for i in range(num_pages):
        bn = region_ID + "oa" + str(i * 30) + "-"
        url = first_url.replace(region_ID, bn)
        list_urls.append(url)
    return list_urls


def hotel_urls(list_urls):
    list_hotel = []
    for page in list_urls:
        result = requests.get(page)
        content = result.content
        soup = BeautifulSoup(content, "html.parser")

        urls = soup.find_all("a", attrs={"class": "review_count"})
        names = soup.find_all("a", attrs={"class": "property_title prominent "})
        assert len(urls) == len(names)
        for i, d in enumerate(urls):
            name = names[i].get_text(strip=True)
            rev_num = int(d.get_text(strip=True).split()[0].replace(',', ''))
            url = d.get("href")
            list_hotel.append((name, url, rev_num))
    list_hotel.sort(key=lambda x: x[2], reverse=True)
    return list_hotel


def review_urls(hotel_link):
    result = requests.get(hotel_link)
    content = result.content
    soup = BeautifulSoup(content, "html.parser")
    num_pages = int(soup.find_all("a", attrs={"class":"pageNum last taLnk "})[0].get_text(strip=True))
    list_urls = []
    for i in range(num_pages):
        bn = "-Reviews-or" + str(i * 5) + "-"
        url = hotel_link.replace("-Reviews-", bn)
        list_urls.append(url)
    return list_urls


def review_urls(hotel_link):
    result = requests.get(hotel_link)
    content = result.content
    soup = BeautifulSoup(content, "html.parser")
    num_pages = int(soup.find_all("a", attrs={"class":"pageNum last taLnk "})[0].get_text(strip=True))
    list_urls = []
    for i in range(num_pages):
        bn = "-Reviews-or" + str(i * 5) + "-"
        url = hotel_link.replace("-Reviews-", bn)
        list_urls.append(url)
    return list_urls


def extract_1page_rev(review_link):
    opts = Options()
    opts.set_headless()
    opts.set_preference("permissions.default.image", 2)
    assert opts.headless  # Operating in headless mode
    reviews = []
    dates = []
    ratings = []
    now = time.time()
    browser = webdriver.Firefox(options=opts)
    # browser = webdriver.Firefox()
    browser.implicitly_wait(10)
    browser.get(review_link)
    review_zone = browser.find_element_by_id("taplc_location_reviews_list_resp_hr_resp_0")
    # find the first button
    more_buttons = review_zone.find_elements_by_css_selector("div.prw_rup.prw_reviews_text_summary_hsx > div > p > span")
    if len(more_buttons) > 0:
        more_buttons[0].click()
        time.sleep(.1)

    revs = review_zone.find_elements_by_css_selector("div.ui_column.is-9 > div.prw_rup.prw_reviews_text_summary_hsx > div > p")
    dats = review_zone.find_elements_by_css_selector("div.ui_column.is-9 > span.ratingDate")
    rats = review_zone.find_elements_by_css_selector("div.ui_column.is-9 > span.ui_bubble_rating")
    assert len(dats) == len(rats) == len(revs)
    for i, rev in enumerate(revs):
        reviews.append(rev.text)
        ratings.append(rats[i].get_attribute("class").split()[-1].split("_")[-1])
        dates.append(dats[i].get_attribute("title"))
    browser.quit()
    print("Streaming %d reviews took %.4f (seconds)" % (len(reviews), time.time()-now))
    return reviews, ratings, dates


def exact_all_revs(filename, review_link, min_page=-1):
    opts = Options()
    opts.set_headless()
    opts.set_preference("permissions.default.image", 2)
    assert opts.headless  # Operating in headless mode
    # browser = webdriver.Firefox(options=opts)
    browser = webdriver.Firefox()
    wait = WebDriverWait(browser, 10)
    browser.implicitly_wait(10)
    browser.get(review_link)
    review_zone = browser.find_element_by_id("taplc_location_reviews_list_resp_hr_resp_0")
    # find the first button
    wait.until_not(EC.visibility_of_element_located((By.CLASS_NAME, "tabs_pers_titles")))
    more_buttons = review_zone.find_elements_by_css_selector(
        "div.prw_rup.prw_reviews_text_summary_hsx > div > p > span")
    if len(more_buttons) > 0:
        more_buttons[0].click()
        time.sleep(.1)
    # review_zone.find_element_by_css_selector("div.prw_rup.prw_reviews_text_summary_hsx > div > p > span").click()
    wait.until(EC.visibility_of_all_elements_located(
        (By.CSS_SELECTOR, 'div.ui_column.is-9 > div.prw_rup.prw_reviews_text_summary_hsx > div > p')))
    data = []
    revs = review_zone.find_elements_by_css_selector(
        "div.ui_column.is-9 > div.prw_rup.prw_reviews_text_summary_hsx > div > p")
    dats = review_zone.find_elements_by_css_selector("div.ui_column.is-9 > span.ratingDate")
    rats = review_zone.find_elements_by_css_selector("div.ui_column.is-9 > span.ui_bubble_rating")
    assert len(dats) == len(rats) == len(revs)
    for j, rev in enumerate(revs):
        data.append(
            (dats[j].get_attribute("title"), rev.text, rats[j].get_attribute("class").split()[-1].split("_")[-1]))
    print("- STREAMING REVIEWS:")
    print("\t+ Pulling page 1: %d reviews" % len(data))
    write_csv_a_lines(filename, data)

    num_pages = int(review_zone.find_element_by_css_selector("a.pageNum.last.taLnk").text)
    if min_page > 0:
        num_pages = min_page

    c = len(data)
    i = 1
    while i < num_pages:
        i += 1
        # located next button and clicked on it
        review_zone.find_element_by_css_selector("a.nav.next.taLnk.ui_button.primary").click()
        # wait until the new content is loaded
        try:
            wait.until_not(EC.visibility_of_element_located((By.ID, "taplc_hotels_loading_box_hr_resp_0")))
        except:
            print("Page %d is reloaded" % i - 1)
            time.sleep(1)

        try:
            wait.until(EC.visibility_of_all_elements_located((By.CSS_SELECTOR,
                                                              'div.ui_column.is-9 > div.prw_rup.prw_reviews_text_summary_hsx > div > p > span')))
            more_buttons = review_zone.find_elements_by_css_selector(
                "div.prw_rup.prw_reviews_text_summary_hsx > div > p > span")
            if len(more_buttons) > 0:
                more_buttons[0].click()
                time.sleep(.1)
        # review_zone.find_element_by_css_selector("div.prw_rup.prw_reviews_text_summary_hsx > div > p > span").click()
        except:
            # In case of not having a <more> field
            print("Page %d doesn't have the 'more' section" % i)
            time.sleep(1)

        try:
            wait.until(EC.visibility_of_all_elements_located(
                (By.CSS_SELECTOR, 'div.ui_column.is-9 > div.prw_rup.prw_reviews_text_summary_hsx > div > p')))
            data = []
            revs = review_zone.find_elements_by_css_selector(
                "div.ui_column.is-9 > div.prw_rup.prw_reviews_text_summary_hsx > div > p")
            dats = review_zone.find_elements_by_css_selector("div.ui_column.is-9 > span.ratingDate")
            rats = review_zone.find_elements_by_css_selector("div.ui_column.is-9 > span.ui_bubble_rating")
            assert len(dats) == len(rats) == len(revs)
            for j, rev in enumerate(revs):
                data.append((dats[j].get_attribute("title"), rev.text,
                             rats[j].get_attribute("class").split()[-1].split("_")[-1]))

            c += len(data)
            print("\t+ Pulling page %d: %d reviews" % (i, c))
            write_csv_a_lines(filename, data)

        except:
            # In case of not having a <more> field
            print("Page %d doesn't have the 'p' section" % i)
            time.sleep(1)
            data = []
            revs = review_zone.find_elements_by_css_selector(
                "div.ui_column.is-9 > div.prw_rup.prw_reviews_text_summary_hsx > div > p")
            dats = review_zone.find_elements_by_css_selector("div.ui_column.is-9 > span.ratingDate")
            rats = review_zone.find_elements_by_css_selector("div.ui_column.is-9 > span.ui_bubble_rating")
            assert len(dats) == len(rats) == len(revs)
            for j, rev in enumerate(revs):
                data.append((dats[j].get_attribute("title"), rev.text,
                             rats[j].get_attribute("class").split()[-1].split("_")[-1]))
            c += len(data)
            print("\t+ Pulling page %d: %d reviews" % (i, c))
            write_csv_a_lines(filename, data)
    browser.quit()


# tripca = "https://www.tripadvisor.ca"
# region_link = "https://www.tripadvisor.ca/Hotels-g154913-Calgary_Alberta-Hotels.html"
# region_ID = "-g154913-"
# region_links = region_urls(region_link)
# hotel_links = hotel_urls(region_links)
# hotel_link = tripca + hotel_links[0][1]
# review_links = review_urls(hotel_link)
# hotel_link = tripca + hotel_links[0][1]
# review_links = review_urls(hotel_link)
# review_link = review_links[0]
# data = extract_1page_rev(review_links[0])
if __name__ == "__main__":
    """
    python trip_review_crawler.py --trip_link https://www.tripadvisor.ca/Hotel_Review-g154913-d183509-Reviews-Sheraton_Suites_Calgary_Eau_Claire-Calgary_Alberta.html --min_page 10
    """
    import os
    import argparse

    argparser = argparse.ArgumentParser()

    argparser.add_argument('--trip_link', help='tripadvisor link',
                           default="https://www.tripadvisor.ca/Hotel_Review-g181808-d579389-Reviews-Holiday_Inn_Express_Suites_Airdrie-Airdrie_Alberta.html",
                           type=str)

    # trip_holiday_link = "https://www.tripadvisor.ca/Hotel_Review-g181808-d579389-Reviews-Holiday_Inn_Express_Suites_Airdrie-Airdrie_Alberta.html"
    # trip_hampton_link = "https://www.tripadvisor.ca/Hotel_Review-g181808-d7332235-Reviews-Hampton_Inn_Suites_Airdrie-Airdrie_Alberta.html"

    argparser.add_argument('--min_page', help='min_page threshold', default=-1, type=int)

    args = argparser.parse_args()

    basename = os.path.basename(args.trip_link)
    filename = "/Users/duytinvo/Projects/aspectSA/hotel/data/customer_reviews/" + basename + ".csv"
    exact_all_revs(filename, args.trip_link, min_page=args.min_page)


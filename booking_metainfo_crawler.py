"""
Created on 2018-09-19
@author: duytinvo
"""
from bs4 import BeautifulSoup
import requests
import time
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


def review_rp(hotel_link, ccode="ca"):
    return hotel_link.split("?")[0].replace("/hotel/%s/" % ccode, "/reviews/%s/hotel/" % ccode)


def url1page(region_link, booking_link, ccode="ca"):
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
            if no_revs >= 1:
                rvurl = review_rp(hurl, ccode)
                hurl = booking_link + hurl
                rvurl = booking_link + rvurl
                hotel_links.append((hname, no_revs, hurl, rvurl))

    nxtbtns = soup.find_all("a", attrs={"title": "Next page"})
    nxt_link = ""
    if len(nxtbtns) == 1:
        nxt_link = nxtbtns[0].get("href")
    return hotel_links, nxt_link


def hotel_urls(region_file, region_link, booking_link, max_page=5, ccode="ca"):
    c = 0
    list_hotel, nxt_link = url1page(region_link, booking_link, ccode)
    while len(nxt_link) != 0:
        # print(nxt_link)
        if max_page > 0:
            c += 1
            if c == max_page:
                break
        hotels, nxt_link = url1page(nxt_link, booking_link, ccode)
        list_hotel.extend(hotels)
    list_hotel.sort(key=lambda x: x[1], reverse=True)
    # TODO: write to a file instead of using list
    joblib.dump(list_hotel, region_file)
    print("\t+ Number of hotels: %d" % len(list_hotel))
    return list_hotel


booking_canada_hotel = {
    "Ontario": ("https://www.booking.com/searchresults.html?label=gen173nr-1DCAEoggJCAlhYSDNYBGgniAEBmAExuAEHyAEP2AED6AEB-AECkgIBeagCAw&sid=b5e75395288459d86dce8ee5ff5808ff&sb=1&src=searchresults&src_elem=sb&error_url=https%3A%2F%2Fwww.booking.com%2Fsearchresults.html%3Flabel%3Dgen173nr-1DCAEoggJCAlhYSDNYBGgniAEBmAExuAEHyAEP2AED6AEB-AECkgIBeagCAw%3Bsid%3Db5e75395288459d86dce8ee5ff5808ff%3Bclass_interval%3D1%3Bdest_id%3D3131%3Bdest_type%3Dregion%3Bfrom_sf%3D1%3Bgroup_adults%3D2%3Bgroup_children%3D0%3Blabel_click%3Dundef%3Bmap%3D1%3Bno_rooms%3D1%3Boffset%3D0%3Braw_dest_type%3Dregion%3Broom1%3DA%252CA%3Bsb_price_type%3Dtotal%3Bsearch_selected%3D1%3Bsrc%3Dindex%3Bsrc_elem%3Dsb%3Bsrpvid%3Deaff7d8a646d0022%3Bss%3DAlberta%252C%2520Canada%3Bss_raw%3Dalberta%3Bssb%3Dempty%26%3B&ss=Ontario%2C+Canada&ssne=Alberta&ssne_untouched=Alberta&checkin_month=&checkin_monthday=&checkin_year=&checkout_month=&checkout_monthday=&checkout_year=&no_rooms=1&group_adults=2&group_children=0&b_h4u_keep_filters=&from_sf=1&ss_raw=onta&ac_position=1&ac_langcode=en&dest_id=659&dest_type=region&place_id_lat=44.13311&place_id_lon=-79.535998&search_pageview_id=eaff7d8a646d0022&search_selected=true&search_pageview_id=eaff7d8a646d0022&ac_suggestion_list_length=5&ac_suggestion_theme_list_length=0",
                "/media/data/hotels/booking_v2/raw_data/booking_ontario_info.pkl"),

    "Quebec": ("https://www.booking.com/searchresults.html?label=gen173nr-1DCAEoggJCAlhYSDNYBGgniAEBmAExuAEHyAEP2AED6AEB-AECkgIBeagCAw&sid=b5e75395288459d86dce8ee5ff5808ff&sb=1&src=searchresults&src_elem=sb&error_url=https%3A%2F%2Fwww.booking.com%2Fsearchresults.html%3Flabel%3Dgen173nr-1DCAEoggJCAlhYSDNYBGgniAEBmAExuAEHyAEP2AED6AEB-AECkgIBeagCAw%3Bsid%3Db5e75395288459d86dce8ee5ff5808ff%3Bclass_interval%3D1%3Bdest_id%3D659%3Bdest_type%3Dregion%3Bdtdisc%3D0%3Bfrom_sf%3D1%3Bgroup_adults%3D2%3Bgroup_children%3D0%3Binac%3D0%3Bindex_postcard%3D0%3Blabel_click%3Dundef%3Bno_rooms%3D1%3Boffset%3D0%3Bpostcard%3D0%3Braw_dest_type%3Dregion%3Broom1%3DA%252CA%3Bsb_price_type%3Dtotal%3Bsearch_selected%3D1%3Bsrc%3Dsearchresults%3Bsrc_elem%3Dsb%3Bsrpvid%3D5ea47dac6a0a008f%3Bss%3DOntario%252C%2520Canada%3Bss_all%3D0%3Bss_raw%3Donta%3Bssb%3Dempty%3Bsshis%3D0%3Bssne_untouched%3DAlberta%26%3B&ss=Quebec%2C+Canada&ssne=Ontario&ssne_untouched=Ontario&checkin_month=&checkin_monthday=&checkin_year=&checkout_month=&checkout_monthday=&checkout_year=&no_rooms=1&group_adults=2&group_children=0&b_h4u_keep_filters=&from_sf=1&ss_raw=quebec&ac_position=2&ac_langcode=en&dest_id=660&dest_type=region&place_id_lat=46.552822&place_id_lon=-72.129969&search_pageview_id=5ea47dac6a0a008f&search_selected=true&search_pageview_id=5ea47dac6a0a008f&ac_suggestion_list_length=5&ac_suggestion_theme_list_length=0",
               "/media/data/hotels/booking_v2/raw_data/booking_quebec_info.pkl"),

    "Nova Scotia": ("https://www.booking.com/searchresults.html?label=gen173nr-1DCAEoggJCAlhYSDNYBGgniAEBmAExuAEHyAEP2AED6AEB-AECkgIBeagCAw&sid=b5e75395288459d86dce8ee5ff5808ff&sb=1&src=searchresults&src_elem=sb&error_url=https%3A%2F%2Fwww.booking.com%2Fsearchresults.html%3Flabel%3Dgen173nr-1DCAEoggJCAlhYSDNYBGgniAEBmAExuAEHyAEP2AED6AEB-AECkgIBeagCAw%3Bsid%3Db5e75395288459d86dce8ee5ff5808ff%3Bclass_interval%3D1%3Bdest_id%3D660%3Bdest_type%3Dregion%3Bdtdisc%3D0%3Bfrom_sf%3D1%3Bgroup_adults%3D2%3Bgroup_children%3D0%3Binac%3D0%3Bindex_postcard%3D0%3Blabel_click%3Dundef%3Bno_rooms%3D1%3Boffset%3D0%3Bpostcard%3D0%3Braw_dest_type%3Dregion%3Broom1%3DA%252CA%3Bsb_price_type%3Dtotal%3Bsearch_selected%3D1%3Bsrc%3Dsearchresults%3Bsrc_elem%3Dsb%3Bsrpvid%3Dacac7dbb598c04a7%3Bss%3DQuebec%252C%2520Canada%3Bss_all%3D0%3Bss_raw%3Dquebec%3Bssb%3Dempty%3Bsshis%3D0%3Bssne_untouched%3DOntario%26%3B&ss=Nova+Scotia%2C+Canada&ssne=Quebec&ssne_untouched=Quebec&checkin_month=&checkin_monthday=&checkin_year=&checkout_month=&checkout_monthday=&checkout_year=&no_rooms=1&group_adults=2&group_children=0&b_h4u_keep_filters=&from_sf=1&ss_raw=Nova+Scotia&ac_position=0&ac_langcode=en&dest_id=661&dest_type=region&place_id_lat=45.153172&place_id_lon=-63.162872&search_pageview_id=acac7dbb598c04a7&search_selected=true&search_pageview_id=acac7dbb598c04a7&ac_suggestion_list_length=5&ac_suggestion_theme_list_length=0",
                   "/media/data/hotels/booking_v2/raw_data/booking_novascotia_info.pkl"),

    "New Brunswick": ("https://www.booking.com/searchresults.html?label=gen173nr-1DCAEoggJCAlhYSDNYBGgniAEBmAExuAEHyAEP2AED6AEB-AECkgIBeagCAw&sid=b5e75395288459d86dce8ee5ff5808ff&sb=1&src=searchresults&src_elem=sb&error_url=https%3A%2F%2Fwww.booking.com%2Fsearchresults.html%3Flabel%3Dgen173nr-1DCAEoggJCAlhYSDNYBGgniAEBmAExuAEHyAEP2AED6AEB-AECkgIBeagCAw%3Bsid%3Db5e75395288459d86dce8ee5ff5808ff%3Bclass_interval%3D1%3Bdest_id%3D661%3Bdest_type%3Dregion%3Bdtdisc%3D0%3Bfrom_sf%3D1%3Bgroup_adults%3D2%3Bgroup_children%3D0%3Binac%3D0%3Bindex_postcard%3D0%3Blabel_click%3Dundef%3Bno_rooms%3D1%3Boffset%3D0%3Bpostcard%3D0%3Braw_dest_type%3Dregion%3Broom1%3DA%252CA%3Bsb_price_type%3Dtotal%3Bsearch_selected%3D1%3Bsrc%3Dsearchresults%3Bsrc_elem%3Dsb%3Bsrpvid%3Dfc457dffb7cc016f%3Bss%3DNova%2520Scotia%252C%2520Canada%3Bss_all%3D0%3Bss_raw%3DNova%2520Scotia%3Bssb%3Dempty%3Bsshis%3D0%3Bssne_untouched%3DQuebec%26%3B&ss=New+Brunswick%2C+Canada&ssne=Nova+Scotia&ssne_untouched=Nova+Scotia&checkin_month=&checkin_monthday=&checkin_year=&checkout_month=&checkout_monthday=&checkout_year=&no_rooms=1&group_adults=2&group_children=0&b_h4u_keep_filters=&from_sf=1&ss_raw=New+Brunswick&ac_position=1&ac_langcode=en&dest_id=3135&dest_type=region&place_id_lat=46.201529&place_id_lon=-65.846068&search_pageview_id=fc457dffb7cc016f&search_selected=true&search_pageview_id=fc457dffb7cc016f&ac_suggestion_list_length=5&ac_suggestion_theme_list_length=0",
                      "/media/data/hotels/booking_v2/raw_data/booking_newbrunswick_info.pkl"),

    "Manitoba": ("https://www.booking.com/searchresults.html?label=gen173nr-1DCAEoggJCAlhYSDNYBGgniAEBmAExuAEHyAEP2AED6AEB-AECkgIBeagCAw&sid=b5e75395288459d86dce8ee5ff5808ff&sb=1&src=searchresults&src_elem=sb&error_url=https%3A%2F%2Fwww.booking.com%2Fsearchresults.html%3Flabel%3Dgen173nr-1DCAEoggJCAlhYSDNYBGgniAEBmAExuAEHyAEP2AED6AEB-AECkgIBeagCAw%3Bsid%3Db5e75395288459d86dce8ee5ff5808ff%3Bclass_interval%3D1%3Bdest_id%3D3135%3Bdest_type%3Dregion%3Bdtdisc%3D0%3Bfrom_sf%3D1%3Bgroup_adults%3D2%3Bgroup_children%3D0%3Binac%3D0%3Bindex_postcard%3D0%3Blabel_click%3Dundef%3Bno_rooms%3D1%3Boffset%3D0%3Bpostcard%3D0%3Braw_dest_type%3Dregion%3Broom1%3DA%252CA%3Bsb_price_type%3Dtotal%3Bsearch_selected%3D1%3Bsrc%3Dsearchresults%3Bsrc_elem%3Dsb%3Bsrpvid%3De5727e2bd49e00c1%3Bss%3DNew%2520Brunswick%252C%2520Canada%3Bss_all%3D0%3Bss_raw%3DNew%2520Brunswick%3Bssb%3Dempty%3Bsshis%3D0%3Bssne_untouched%3DNova%2520Scotia%26%3B&ss=Manitoba%2C+Canada&ssne=New+Brunswick&ssne_untouched=New+Brunswick&checkin_month=&checkin_monthday=&checkin_year=&checkout_month=&checkout_monthday=&checkout_year=&no_rooms=1&group_adults=2&group_children=0&b_h4u_keep_filters=&from_sf=1&ss_raw=Manitoba&ac_position=0&ac_langcode=en&dest_id=3133&dest_type=region&place_id_lat=50.546447&place_id_lon=-97.993978&search_pageview_id=e5727e2bd49e00c1&search_selected=true&search_pageview_id=e5727e2bd49e00c1&ac_suggestion_list_length=5&ac_suggestion_theme_list_length=0",
                 "/media/data/hotels/booking_v2/raw_data/booking_manitoba_info.pkl"),

    "British Columbia": ("https://www.booking.com/searchresults.html?label=gen173nr-1DCAEoggJCAlhYSDNYBGgniAEBmAExuAEHyAEP2AED6AEB-AECkgIBeagCAw&sid=b5e75395288459d86dce8ee5ff5808ff&sb=1&src=searchresults&src_elem=sb&error_url=https%3A%2F%2Fwww.booking.com%2Fsearchresults.html%3Flabel%3Dgen173nr-1DCAEoggJCAlhYSDNYBGgniAEBmAExuAEHyAEP2AED6AEB-AECkgIBeagCAw%3Bsid%3Db5e75395288459d86dce8ee5ff5808ff%3Bclass_interval%3D1%3Bdest_id%3D3133%3Bdest_type%3Dregion%3Bdtdisc%3D0%3Bfrom_sf%3D1%3Bgroup_adults%3D2%3Bgroup_children%3D0%3Binac%3D0%3Bindex_postcard%3D0%3Blabel_click%3Dundef%3Bno_rooms%3D1%3Boffset%3D0%3Bpostcard%3D0%3Braw_dest_type%3Dregion%3Broom1%3DA%252CA%3Bsb_price_type%3Dtotal%3Bsearch_selected%3D1%3Bsrc%3Dsearchresults%3Bsrc_elem%3Dsb%3Bsrpvid%3D60767e4da27b0023%3Bss%3DManitoba%252C%2520Canada%3Bss_all%3D0%3Bss_raw%3DManitoba%3Bssb%3Dempty%3Bsshis%3D0%3Bssne_untouched%3DNew%2520Brunswick%26%3B&ss=British+Columbia%2C+Canada&ssne=Manitoba&ssne_untouched=Manitoba&checkin_month=&checkin_monthday=&checkin_year=&checkout_month=&checkout_monthday=&checkout_year=&no_rooms=1&group_adults=2&group_children=0&b_h4u_keep_filters=&from_sf=1&ss_raw=British+Columbia&ac_position=0&ac_langcode=en&dest_id=658&dest_type=region&place_id_lat=50.00405&place_id_lon=-121.983124&search_pageview_id=60767e4da27b0023&search_selected=true&search_pageview_id=60767e4da27b0023&ac_suggestion_list_length=5&ac_suggestion_theme_list_length=0",
                         "/media/data/hotels/booking_v2/raw_data/booking_britishcolumbia_info.pkl"),

    "Prince Edward Island": ("https://www.booking.com/searchresults.html?label=gen173nr-1DCAEoggJCAlhYSDNYBGgniAEBmAExuAEHyAEP2AED6AEB-AECkgIBeagCAw&sid=b5e75395288459d86dce8ee5ff5808ff&sb=1&src=searchresults&src_elem=sb&error_url=https%3A%2F%2Fwww.booking.com%2Fsearchresults.html%3Flabel%3Dgen173nr-1DCAEoggJCAlhYSDNYBGgniAEBmAExuAEHyAEP2AED6AEB-AECkgIBeagCAw%3Bsid%3Db5e75395288459d86dce8ee5ff5808ff%3Bclass_interval%3D1%3Bdest_id%3D658%3Bdest_type%3Dregion%3Bdtdisc%3D0%3Bfrom_sf%3D1%3Bgroup_adults%3D2%3Bgroup_children%3D0%3Binac%3D0%3Bindex_postcard%3D0%3Blabel_click%3Dundef%3Bno_rooms%3D1%3Boffset%3D0%3Bpostcard%3D0%3Braw_dest_type%3Dregion%3Broom1%3DA%252CA%3Bsb_price_type%3Dtotal%3Bsearch_selected%3D1%3Bsrc%3Dsearchresults%3Bsrc_elem%3Dsb%3Bsrpvid%3D6adf7e65fa1e0335%3Bss%3DBritish%2520Columbia%252C%2520Canada%3Bss_all%3D0%3Bss_raw%3DBritish%2520Columbia%3Bssb%3Dempty%3Bsshis%3D0%3Bssne_untouched%3DManitoba%26%3B&ss=Prince+Edward+Island%2C+Canada&ssne=British+Columbia&ssne_untouched=British+Columbia&checkin_month=&checkin_monthday=&checkin_year=&checkout_month=&checkout_monthday=&checkout_year=&no_rooms=1&group_adults=2&group_children=0&b_h4u_keep_filters=&from_sf=1&ss_raw=Prince+Edward+Island&ac_position=0&ac_langcode=en&dest_id=3137&dest_type=region&place_id_lat=46.336144&place_id_lon=-63.21678&search_pageview_id=6adf7e65fa1e0335&search_selected=true&search_pageview_id=6adf7e65fa1e0335&ac_suggestion_list_length=5&ac_suggestion_theme_list_length=0",
                             "/media/data/hotels/booking_v2/raw_data/booking_princeedwardisland_info.pkl"),

    "Saskatchewan": ("https://www.booking.com/searchresults.html?label=gen173nr-1DCAEoggJCAlhYSDNYBGgniAEBmAExuAEHyAEP2AED6AEB-AECkgIBeagCAw&sid=b5e75395288459d86dce8ee5ff5808ff&sb=1&src=searchresults&src_elem=sb&error_url=https%3A%2F%2Fwww.booking.com%2Fsearchresults.html%3Flabel%3Dgen173nr-1DCAEoggJCAlhYSDNYBGgniAEBmAExuAEHyAEP2AED6AEB-AECkgIBeagCAw%3Bsid%3Db5e75395288459d86dce8ee5ff5808ff%3Bclass_interval%3D1%3Bdest_id%3D3137%3Bdest_type%3Dregion%3Bdtdisc%3D0%3Bfrom_sf%3D1%3Bgroup_adults%3D2%3Bgroup_children%3D0%3Binac%3D0%3Bindex_postcard%3D0%3Blabel_click%3Dundef%3Bno_rooms%3D1%3Boffset%3D0%3Bpostcard%3D0%3Braw_dest_type%3Dregion%3Broom1%3DA%252CA%3Bsb_price_type%3Dtotal%3Bsearch_selected%3D1%3Bsrc%3Dsearchresults%3Bsrc_elem%3Dsb%3Bsrpvid%3Ddb617e8d30df00fd%3Bss%3DPrince%2520Edward%2520Island%252C%2520Canada%3Bss_all%3D0%3Bss_raw%3DPrince%2520Edward%2520Island%3Bssb%3Dempty%3Bsshis%3D0%3Bssne_untouched%3DBritish%2520Columbia%26%3B&ss=Saskatchewan%2C+Canada&ssne=Prince+Edward+Island&ssne_untouched=Prince+Edward+Island&checkin_month=&checkin_monthday=&checkin_year=&checkout_month=&checkout_monthday=&checkout_year=&no_rooms=1&group_adults=2&group_children=0&b_h4u_keep_filters=&from_sf=1&ss_raw=Saskatchewan&ac_position=0&ac_langcode=en&dest_id=3134&dest_type=region&place_id_lat=51.349954&place_id_lon=-105.567444&search_pageview_id=db617e8d30df00fd&search_selected=true&search_pageview_id=db617e8d30df00fd&ac_suggestion_list_length=5&ac_suggestion_theme_list_length=0",
                     "/media/data/hotels/booking_v2/raw_data/booking_saskatchewan_info.pkl"),

    "Alberta": ("https://www.booking.com/searchresults.html?label=gen173nr-1DCAEoggJCAlhYSDNYBGgniAEBmAExuAEHyAEP2AED6AEB-AECkgIBeagCAw&sid=b5e75395288459d86dce8ee5ff5808ff&sb=1&src=searchresults&src_elem=sb&error_url=https%3A%2F%2Fwww.booking.com%2Fsearchresults.html%3Flabel%3Dgen173nr-1DCAEoggJCAlhYSDNYBGgniAEBmAExuAEHyAEP2AED6AEB-AECkgIBeagCAw%3Bsid%3Db5e75395288459d86dce8ee5ff5808ff%3Bclass_interval%3D1%3Bdest_id%3D3134%3Bdest_type%3Dregion%3Bdtdisc%3D0%3Bfrom_sf%3D1%3Bgroup_adults%3D2%3Bgroup_children%3D0%3Binac%3D0%3Bindex_postcard%3D0%3Blabel_click%3Dundef%3Bno_rooms%3D1%3Boffset%3D0%3Bpostcard%3D0%3Braw_dest_type%3Dregion%3Broom1%3DA%252CA%3Bsb_price_type%3Dtotal%3Bsearch_selected%3D1%3Bsrc%3Dsearchresults%3Bsrc_elem%3Dsb%3Bsrpvid%3Df3847eb0748702ee%3Bss%3DSaskatchewan%252C%2520Canada%3Bss_all%3D0%3Bss_raw%3DSaskatchewan%3Bssb%3Dempty%3Bsshis%3D0%3Bssne_untouched%3DPrince%2520Edward%2520Island%26%3B&ss=Alberta%2C+Canada&ssne=Saskatchewan&ssne_untouched=Saskatchewan&checkin_month=&checkin_monthday=&checkin_year=&checkout_month=&checkout_monthday=&checkout_year=&no_rooms=1&group_adults=2&group_children=0&b_h4u_keep_filters=&from_sf=1&ss_raw=Alberta&ac_position=0&ac_langcode=en&dest_id=3131&dest_type=region&place_id_lat=52.351655&place_id_lon=-114.202013&search_pageview_id=f3847eb0748702ee&search_selected=true&search_pageview_id=f3847eb0748702ee&ac_suggestion_list_length=5&ac_suggestion_theme_list_length=0",
                "/media/data/hotels/booking_v2/raw_data/booking_alberta_info.pkl"),

    "Newfoundland and Labrador": ("https://www.booking.com/searchresults.html?label=gen173nr-1DCAEoggJCAlhYSDNYBGgniAEBmAExuAEHyAEP2AED6AEB-AECkgIBeagCAw&sid=b5e75395288459d86dce8ee5ff5808ff&sb=1&src=searchresults&src_elem=sb&error_url=https%3A%2F%2Fwww.booking.com%2Fsearchresults.html%3Flabel%3Dgen173nr-1DCAEoggJCAlhYSDNYBGgniAEBmAExuAEHyAEP2AED6AEB-AECkgIBeagCAw%3Bsid%3Db5e75395288459d86dce8ee5ff5808ff%3Bclass_interval%3D1%3Bdest_id%3D3131%3Bdest_type%3Dregion%3Bdtdisc%3D0%3Bfrom_sf%3D1%3Bgroup_adults%3D2%3Bgroup_children%3D0%3Binac%3D0%3Bindex_postcard%3D0%3Blabel_click%3Dundef%3Bno_rooms%3D1%3Boffset%3D0%3Bpostcard%3D0%3Braw_dest_type%3Dregion%3Broom1%3DA%252CA%3Bsb_price_type%3Dtotal%3Bsearch_selected%3D1%3Bsrc%3Dsearchresults%3Bsrc_elem%3Dsb%3Bsrpvid%3D863e7ede48210044%3Bss%3DAlberta%252C%2520Canada%3Bss_all%3D0%3Bss_raw%3DAlberta%3Bssb%3Dempty%3Bsshis%3D0%3Bssne_untouched%3DSaskatchewan%26%3B&ss=Newfoundland+and+Labrador%2C+Canada&ssne=Alberta&ssne_untouched=Alberta&checkin_month=&checkin_monthday=&checkin_year=&checkout_month=&checkout_monthday=&checkout_year=&no_rooms=1&group_adults=2&group_children=0&b_h4u_keep_filters=&from_sf=1&ss_raw=Newfoundland+and+Labrador&ac_position=0&ac_langcode=en&dest_id=3136&dest_type=region&place_id_lat=48.461745&place_id_lon=-54.4745&search_pageview_id=863e7ede48210044&search_selected=true&search_pageview_id=863e7ede48210044&ac_suggestion_list_length=5&ac_suggestion_theme_list_length=0",
                                  "/media/data/hotels/booking_v2/raw_data/booking_newfoundlandlabrador_info.pkl"),

    "Northwest Territories": ("https://www.booking.com/searchresults.html?label=gen173nr-1DCAEoggJCAlhYSDNYBGgniAEBmAExuAEHyAEP2AED6AEB-AECkgIBeagCAw&sid=b5e75395288459d86dce8ee5ff5808ff&sb=1&src=searchresults&src_elem=sb&error_url=https%3A%2F%2Fwww.booking.com%2Fsearchresults.html%3Flabel%3Dgen173nr-1DCAEoggJCAlhYSDNYBGgniAEBmAExuAEHyAEP2AED6AEB-AECkgIBeagCAw%3Bsid%3Db5e75395288459d86dce8ee5ff5808ff%3Bclass_interval%3D1%3Bdest_id%3D3136%3Bdest_type%3Dregion%3Bdtdisc%3D0%3Bfrom_sf%3D1%3Bgroup_adults%3D2%3Bgroup_children%3D0%3Binac%3D0%3Bindex_postcard%3D0%3Blabel_click%3Dundef%3Bno_rooms%3D1%3Boffset%3D0%3Bpostcard%3D0%3Braw_dest_type%3Dregion%3Broom1%3DA%252CA%3Bsb_price_type%3Dtotal%3Bsearch_selected%3D1%3Bsrc%3Dsearchresults%3Bsrc_elem%3Dsb%3Bsrpvid%3D7c4e7effd284026f%3Bss%3DNewfoundland%2520and%2520Labrador%252C%2520Canada%3Bss_all%3D0%3Bss_raw%3DNewfoundland%2520and%2520Labrador%3Bssb%3Dempty%3Bsshis%3D0%3Bssne_untouched%3DAlberta%26%3B&ss=Northwest+Territories%2C+Canada&ssne=Newfoundland+and+Labrador&ssne_untouched=Newfoundland+and+Labrador&checkin_month=&checkin_monthday=&checkin_year=&checkout_month=&checkout_monthday=&checkout_year=&no_rooms=1&group_adults=2&group_children=0&b_h4u_keep_filters=&from_sf=1&ss_raw=Northwest+Territories&ac_position=0&ac_langcode=en&dest_id=3140&dest_type=region&place_id_lat=62.883816&place_id_lon=-116.100309&search_pageview_id=7c4e7effd284026f&search_selected=true&search_pageview_id=7c4e7effd284026f&ac_suggestion_list_length=1&ac_suggestion_theme_list_length=0",
                              "/media/data/hotels/booking_v2/raw_data/booking_northwestterritories_info.pkl"),

    "Yukon": ("https://www.booking.com/searchresults.html?label=gen173nr-1DCAEoggJCAlhYSDNYBGgniAEBmAExuAEHyAEP2AED6AEB-AECkgIBeagCAw&sid=b5e75395288459d86dce8ee5ff5808ff&sb=1&src=searchresults&src_elem=sb&error_url=https%3A%2F%2Fwww.booking.com%2Fsearchresults.html%3Flabel%3Dgen173nr-1DCAEoggJCAlhYSDNYBGgniAEBmAExuAEHyAEP2AED6AEB-AECkgIBeagCAw%3Bsid%3Db5e75395288459d86dce8ee5ff5808ff%3Bclass_interval%3D1%3Bdest_id%3D3140%3Bdest_type%3Dregion%3Bdtdisc%3D0%3Bfrom_sf%3D1%3Bgroup_adults%3D2%3Bgroup_children%3D0%3Binac%3D0%3Bindex_postcard%3D0%3Blabel_click%3Dundef%3Bno_rooms%3D1%3Boffset%3D0%3Bpostcard%3D0%3Braw_dest_type%3Dregion%3Broom1%3DA%252CA%3Bsb_price_type%3Dtotal%3Bsearch_selected%3D1%3Bsrc%3Dsearchresults%3Bsrc_elem%3Dsb%3Bsrpvid%3D3bf27f27a09200a8%3Bss%3DNorthwest%2520Territories%252C%2520Canada%3Bss_all%3D0%3Bss_raw%3DNorthwest%2520Territories%3Bssb%3Dempty%3Bsshis%3D0%3Bssne_untouched%3DNewfoundland%2520and%2520Labrador%26%3B&ss=Yukon%2C+Canada&ssne=Northwest+Territories&ssne_untouched=Northwest+Territories&checkin_month=&checkin_monthday=&checkin_year=&checkout_month=&checkout_monthday=&checkout_year=&no_rooms=1&group_adults=2&group_children=0&b_h4u_keep_filters=&from_sf=1&ss_raw=Yukon&ac_position=1&ac_langcode=en&dest_id=3138&dest_type=region&place_id_lat=61.301707&place_id_lon=-136.136483&search_pageview_id=3bf27f27a09200a8&search_selected=true&search_pageview_id=3bf27f27a09200a8&ac_suggestion_list_length=5&ac_suggestion_theme_list_length=0",
              "/media/data/hotels/booking_v2/raw_data/booking_yukon_info.pkl"),

    "Nunavut": ("https://www.booking.com/searchresults.html?label=gen173nr-1DCAEoggJCAlhYSDNYBGgniAEBmAExuAEHyAEP2AED6AEB-AECkgIBeagCAw&sid=b5e75395288459d86dce8ee5ff5808ff&sb=1&src=searchresults&src_elem=sb&error_url=https%3A%2F%2Fwww.booking.com%2Fsearchresults.html%3Flabel%3Dgen173nr-1DCAEoggJCAlhYSDNYBGgniAEBmAExuAEHyAEP2AED6AEB-AECkgIBeagCAw%3Bsid%3Db5e75395288459d86dce8ee5ff5808ff%3Bclass_interval%3D1%3Bdest_id%3D3138%3Bdest_type%3Dregion%3Bdtdisc%3D0%3Bfrom_sf%3D1%3Bgroup_adults%3D2%3Bgroup_children%3D0%3Binac%3D0%3Bindex_postcard%3D0%3Blabel_click%3Dundef%3Bno_rooms%3D1%3Boffset%3D0%3Bpostcard%3D0%3Braw_dest_type%3Dregion%3Broom1%3DA%252CA%3Bsb_price_type%3Dtotal%3Bsearch_selected%3D1%3Bsrc%3Dsearchresults%3Bsrc_elem%3Dsb%3Bsrpvid%3D22ed7f52d429001f%3Bss%3DYukon%252C%2520Canada%3Bss_all%3D0%3Bss_raw%3DYukon%3Bssb%3Dempty%3Bsshis%3D0%3Bssne_untouched%3DNorthwest%2520Territories%26%3B&ss=Nunavut%2C+Canada&ssne=Yukon&ssne_untouched=Yukon&checkin_month=&checkin_monthday=&checkin_year=&checkout_month=&checkout_monthday=&checkout_year=&no_rooms=1&group_adults=2&group_children=0&b_h4u_keep_filters=&from_sf=1&ss_raw=Nunavut&ac_position=0&ac_langcode=en&dest_id=3139&dest_type=region&place_id_lat=65.537769&place_id_lon=-80.698361&search_pageview_id=22ed7f52d429001f&search_selected=true&search_pageview_id=22ed7f52d429001f&ac_suggestion_list_length=1&ac_suggestion_theme_list_length=0",
                "/media/data/hotels/booking_v2/raw_data/booking_nunavut_info.pkl")}


booking_australia_hotel = {
    "Australian Capital Territory": "https://www.booking.com/searchresults.en-gb.html?label=gen173nr-1DCA0oJ0IFbm9tYWRIM1gEaCeIAQGYAS64AQfIAQzYAQPoAQGSAgF5qAID;sid=b4be965cfe5d1ffcf63265b632204bc8;atlas_src=lp_map;dest_id=1381;dest_type=region;srpvid=2f0085ea41ff0328&;map=1#map_closed",
    "New South Wales": "https://www.booking.com/searchresults.en-gb.html?label=gen173nr-1DCA0oJ0IFbm9tYWRIM1gEaCeIAQGYAS64AQfIAQzYAQPoAQGSAgF5qAID;sid=b4be965cfe5d1ffcf63265b632204bc8;atlas_src=lp_map;dest_id=612;dest_type=region;srpvid=2f0085ea41ff0328&;map=1#map_closed",
    "Northern Territory": "https://www.booking.com/searchresults.en-gb.html?label=gen173nr-1DCA0oJ0IFbm9tYWRIM1gEaCeIAQGYAS64AQfIAQzYAQPoAQGSAgF5qAID;sid=b4be965cfe5d1ffcf63265b632204bc8;atlas_src=lp_map;dest_id=613;dest_type=region;srpvid=2f0085ea41ff0328&;map=1#map_closed",
    "Queensland": "https://www.booking.com/searchresults.en-gb.html?label=gen173nr-1DCA0oJ0IFbm9tYWRIM1gEaCeIAQGYAS64AQfIAQzYAQPoAQGSAgF5qAID;sid=b4be965cfe5d1ffcf63265b632204bc8;atlas_src=lp_map;dest_id=614;dest_type=region;srpvid=2f0085ea41ff0328&;map=1#map_closed",
    "South Australia": "https://www.booking.com/searchresults.en-gb.html?label=gen173nr-1DCA0oJ0IFbm9tYWRIM1gEaCeIAQGYAS64AQfIAQzYAQPoAQGSAgF5qAID;sid=b4be965cfe5d1ffcf63265b632204bc8;atlas_src=lp_map;dest_id=1380;dest_type=region;srpvid=2f0085ea41ff0328&;map=1#map_closed",
    "Tasmania": "https://www.booking.com/searchresults.en-gb.html?label=gen173nr-1DCA0oJ0IFbm9tYWRIM1gEaCeIAQGYAS64AQfIAQzYAQPoAQGSAgF5qAID;sid=b4be965cfe5d1ffcf63265b632204bc8;atlas_src=lp_map;dest_id=616;dest_type=region;srpvid=2f0085ea41ff0328&;map=1#map_closed",
    "Victoria": "https://www.booking.com/searchresults.en-gb.html?label=gen173nr-1DCA0oJ0IFbm9tYWRIM1gEaCeIAQGYAS64AQfIAQzYAQPoAQGSAgF5qAID;sid=b4be965cfe5d1ffcf63265b632204bc8;atlas_src=lp_map;dest_id=617;dest_type=region;srpvid=2f0085ea41ff0328&;map=1#map_closed",
    "Western Australia": "https://www.booking.com/searchresults.en-gb.html?label=gen173nr-1DCA0oJ0IFbm9tYWRIM1gEaCeIAQGYAS64AQfIAQzYAQPoAQGSAgF5qAID;sid=b4be965cfe5d1ffcf63265b632204bc8;atlas_src=lp_map;dest_id=1379;dest_type=region;srpvid=2f0085ea41ff0328&;map=1#map_closed"}


def canada_nation_url():
    booking_link = "https://www.booking.com"
    for region in booking_canada_hotel:
        start = time.time()
        region_link = booking_canada_hotel[region][0]
        region_file= booking_canada_hotel[region][1]
        print("- Extracting hotel urls in '%s' province" % region)
        hotel_links = hotel_urls(region_file, region_link, booking_link, max_page=-1)
        print("\t+ Processing time: %.4f(s)" % (time.time()-start))


def nation_url(national_hotel, path="/media/data/hotels/booking_v3/raw_data/australia/", ccode="ca"):
    booking_link = "https://www.booking.com"
    for region in national_hotel:
        start = time.time()
        region_link = national_hotel[region]
        region_file = path + region.replace(" ", "_").lower() + "_info.pkl"
        print("- Extracting hotel urls in '%s' province" % region)
        hotel_links = hotel_urls(region_file, region_link, booking_link, max_page=-1, ccode=ccode)
        print("\t+ Processing time: %.4f(s)" % (time.time()-start))


if __name__ == "__main__":
    nation_url(booking_australia_hotel, path="/media/data/hotels/booking_v3/raw_data/australia/", ccode="au")

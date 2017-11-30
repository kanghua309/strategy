# -*- coding: utf-8 -*-
"""
Created on Wed May 17 13:19:27 2017

@author: kanghua
"""
import pandas as pd
import numpy as np
import math


import zipline
from zipline.api import (
    symbol,
    sid,
)
from zipline.pipeline.factors import AverageDollarVolume

from me.pipeline.factors.tsfactor import Fundamental
from me.pipeline.factors.boost import ADV_adj
from me.pipeline.classifiers.tushare.sector import get_sector,get_sector_class
from zipline.pipeline.filters import CustomFilter

from me.pipeline.utils.meta import load_tushare_df

MARKET_CAP_DOWNLIMIT = 5.0e+9
ADV_ADJ_DOWNLIMIT = 1.0e+8


def universe_filter(smoothing_func = None):
    """
    Create a Pipeline producing Filters implementing common acceptance criteria.
    Returns
    -------
    zipline.Filter
        Filter to control tradeablility
    """
    factors = {
     'MarketCap': Fundamental().outstanding,
     'ADV_adj'  : ADV_adj(),
     'Sector'   : get_sector(),
    }


    #func = lambda f: f.downsample('month_start')
    if smoothing_func != None:
        factors = { id:smoothing_func(factors[id]) for id in factors.keys() }
    #print factors
    factors['MarketCap'] = factors['MarketCap'] > MARKET_CAP_DOWNLIMIT
    factors['ADV_adj']   = factors['ADV_adj']   > ADV_ADJ_DOWNLIMIT
    factors['Sector']    = factors['Sector'].notnull()

    filters = None
    for value in factors.values():
        if filters == None:
           filters = value
        else:
           filters = (filters & value)

    #print filters

    #mySymbolsListfiter = default_china_equity_universe_mask()
    # Equities with an average daily volume greater than 5000000.
    #high_volume = (AverageDollarVolume(window_length=252) > 5000000)
    #liquid = ADV_adj().downsample('month_start') > 2500000
    #market_cap_filter = MarketCap().downsample('month_start') > market_cap_limit
    #universe_filter = (mySymbolsListfiter & market_cap_filter & liquid & high_volume )
    #universe_filter = ( maket_cap_filter & liquid )

    return filters

def sector_filter(tradeable_count,sector_exposure_limit,smoothing_func = None):

    """
    Mask for Pipeline in create_tradeable. Limits each sector so as not to be over-exposed

    Parameters
    ----------
    tradeable_count : int
        Target number of constituent securities in universe
    sector_exposure_limit: float
        Target threshold for any particular sector
    Returns
    -------
    zipline.Filter
        Filter to control sector exposure
    """

    industry_class = get_sector_class()
    #print("g_inds",g_inds)
    sector_factor = get_sector(industry_class)
    # set thresholds
    sector_size = len(industry_class)
    if sector_exposure_limit < ((1. / sector_size)):
        threshold = int(math.ceil((1. /sector_size) * tradeable_count))
    elif sector_exposure_limit > 1.:
        threshold = tradeable_count
    else:
        threshold = int(math.ceil(sector_exposure_limit * tradeable_count))

    #print ("tradeable_count %s , get_sector_size %s , industry threashold %s:" % (tradeable_count,sector_size,threshold))
    filters = None
    for industry,ino in industry_class.items():
        #print industry,ino,industry_class[industry]
        mask=sector_factor.eq(industry_class[industry])
        if smoothing_func != None:
           value = smoothing_func(AverageDollarVolume(window_length=21)).top(threshold,mask)
        else:
           value = AverageDollarVolume(window_length=21).top(threshold, mask)
        #print value
        #print filters
        if filters == None:
           filters = value
        else:
           filters = (filters | value)
    '''
    transport_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['交通运输'.decode("UTF-8")]))             #
    instrument_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['仪器仪表'.decode("UTF-8")]))
    media_entertainment_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['传媒娱乐'.decode("UTF-8")]))   #
    water_supply_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['供水供气'.decode("UTF-8")]))          #
    highway_bridge_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['公路桥梁'.decode("UTF-8")]))        #
    other_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['其它行业'.decode("UTF-8")]))
    animal_husbandry_fishery_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['农林牧渔'.decode("UTF-8")]))  #
    pesticide_fertilizer_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['农药化肥'.decode("UTF-8")]))
    chemical_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['化工行业'.decode("UTF-8")]))                  #
    chemical_fiber_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['化纤行业'.decode("UTF-8")]))
    medical_device_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['医疗器械'.decode("UTF-8")]))
    printing_packaging_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['印刷包装'.decode("UTF-8")]))
    power_plant_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['发电设备'.decode("UTF-8")]))
    business_department_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['商业百货'.decode("UTF-8")]))       #
    plastics_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['塑料制品'.decode("UTF-8")]))
    furniture_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['家具行业'.decode("UTF-8")]))                 #
    appliance_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['家电行业'.decode("UTF-8")]))                 #
    building_materials_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['建筑建材'.decode("UTF-8")]))        #
    development_zone_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['开发区'.decode("UTF-8")]))
    real_estate_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['房地产'.decode("UTF-8")]))                #
    motorcycle_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['摩托车'.decode("UTF-8")]))
    nonferrous_metals_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['有色金属'.decode("UTF-8")]))         #
    clothing_footwear_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['服装鞋类'.decode("UTF-8")]))         #
    machinery_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['机械行业'.decode("UTF-8")]))                 #
    time_shares_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['次新股'.decode("UTF-8")]))
    cement_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['水泥行业'.decode("UTF-8")]))
    automobile_manufacturing_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['汽车制造'.decode("UTF-8")]))  #
    coal_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['煤炭行业'.decode("UTF-8")]))                      #
    material_trade_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['物资外贸'.decode("UTF-8")]))
    environmental_protection_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['环保行业'.decode("UTF-8")]))   #
    glass_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['玻璃行业'.decode("UTF-8")]))
    biopharmaceuticals_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['生物制药'.decode("UTF-8")]))         #
    power_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['电力行业'.decode("UTF-8")]))                     
    electronic_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['电器行业'.decode("UTF-8")]))
    electronic_information_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['电子信息'.decode("UTF-8")]))     #
    electronic_device_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['电子器件'.decode("UTF-8")]))            
    oil_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['石油行业'.decode("UTF-8")]))                     
    textile_machinery_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['纺织机械'.decode("UTF-8")]))
    textile_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['纺织行业'.decode("UTF-8")]))
    integrated_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['综合行业'.decode("UTF-8")]))
    ship_manufacturing_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['船舶制造'.decode("UTF-8")]))
    paper_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['造纸行业'.decode("UTF-8")]))
    hotel_tour_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['酒店旅游'.decode("UTF-8")]))                #
    wine_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['酿酒行业'.decode("UTF-8")]))
    financial_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['金融行业'.decode("UTF-8")]))                 #
    steel_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['钢铁行业'.decode("UTF-8")]))                     #      
    ceramics_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['陶瓷行业'.decode("UTF-8")]))           
    aircraft_manufacturing_trim = AverageDollarVolume(window_length=21).top(threshold, mask=sector_factor.eq(industry_class['飞机制造'.decode("UTF-8")]))
    food_industry_trim = AverageDollarVolume(window_length=21).downsample('month_start').top(threshold, mask=sector_factor.eq(industry_class['食品行业'.decode("UTF-8")]))             #

    return transport_trim|media_entertainment_trim|\
           chemical_trim|medical_device_trim|\
           power_plant_trim|business_department_trim|appliance_trim|\
           building_materials_trim|real_estate_trim|nonferrous_metals_trim|\
           clothing_footwear_trim|machinery_trim|automobile_manufacturing_trim|\
           coal_trim|environmental_protection_trim|biopharmaceuticals_trim|\
           power_trim|electronic_information_trim|electronic_device_trim|\
           oil_trim|textile_trim|hotel_tour_trim|\
           wine_trim|financial_trim|steel_trim|\
           aircraft_manufacturing_trim|food_industry_trim
    '''
    return filters


def default_china_equity_universe_mask(unmask):
    #a_stocks = []
    info = load_tushare_df("basic")
    sme =  load_tushare_df("sme")
    gem =  load_tushare_df("gem")
    st  =  load_tushare_df("st")
    uset = pd.concat([sme, gem, st])
    maskdf  = info.drop([y for y in uset['code']], axis=0)  # st,sme,gem 的都不要，稳健型只要主板股票
    maskdf = maskdf.drop(unmask,axis=0)
    #Returns a factor indicating membership (=1) in the given iterable of securities
    #print("==enter IsInSymbolsList==")
    class IsInDefaultChinaUniverse(CustomFilter):
        inputs = [];
        window_length = 1
        def compute(self, today, asset_ids, out, *inputs):
            #print asset_ids
            #print maskset
            assets  = [sid(id).symbol for id in asset_ids]
            #print "--------------"
            #print pd.Series(assets)
            out[:] = pd.Series(assets).isin(maskdf.index)
            #print out
    return IsInDefaultChinaUniverse()

def private_universe_mask(mask,asset_finder = None):
    mask = mask
    def sid(sid):
        return asset_finder.retrieve_asset(sid)
    class IsInPrivateUniverse(CustomFilter):
        inputs = [];
        window_length = 1
        def compute(self, today, asset_ids, out, *inputs):
            #print maskset
            #print asset_ids
            # for id in asset_ids:
            #     print id
            #     print type(sid(id))
            assets  = [sid(id).symbol for id in asset_ids]
            #print "--------------"
            #print pd.Series(assets)
            out[:] = pd.Series(assets).isin(mask)
            #print out
    return IsInPrivateUniverse()


def make_china_equity_universe(
        target_size,
        #rankby,
        mask,
        #groupby,
        max_group_weight,
        smoothing_func):
    ufilters = universe_filter(smoothing_func)
    sfilters = sector_filter(target_size,max_group_weight,smoothing_func)

    return (ufilters & sfilters) & mask # &? TODO
    #return (sfilters)



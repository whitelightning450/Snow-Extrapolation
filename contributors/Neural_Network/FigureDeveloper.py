import os
import pandas as pd
import warnings
import hydroeval as he
import random
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from pickle import dump
import pickle 
from tqdm import tqdm
from mpl_toolkits.basemap import Basemap
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import contextily as cx

warnings.filterwarnings("ignore")



def Dict_2_DF(EvalDF, Region_list):
    #Change the location names to improve the interpretability of the figures
    Model_Results= pd.DataFrame(columns = ['y_test', 'y_pred',  'Region'])

    for region in Region_list:
        if region == 'N_Sierras':
            EvalDF[region]['Region'] = 'Northern Sierra Nevada'
        elif region == 'S_Sierras_High':
            EvalDF[region]['Region'] = 'Southern Sierra Nevada High'
        elif region == 'S_Sierras_Low':
            EvalDF[region]['Region'] = 'Southern Sierra Nevada Low'
        elif region == 'Greater_Yellowstone':
            EvalDF[region]['Region'] = 'Greater Yellowstone'
        elif region == 'N_Co_Rockies':
            EvalDF[region]['Region'] = 'Upper Colorado Rockies'
        elif region == 'SW_Mont':
            EvalDF[region]['Region'] = 'SW Montana'
        elif region == 'SW_Co_Rockies':
            EvalDF[region]['Region'] = 'San Juan Mountains'
        elif region == 'GBasin':
            EvalDF[region]['Region'] = 'Great Basin'
        elif region == 'N_Wasatch':
            EvalDF[region]['Region'] = 'Northern Wasatch'
        elif region == 'N_Cascade':
            EvalDF[region]['Region'] = 'Northern Cascades'
        elif region == 'S_Wasatch':
            EvalDF[region]['Region'] = 'SW Utah'
        elif region == 'SW_Mtns':
            EvalDF[region]['Region'] = 'SW Desert'
        elif region == 'E_WA_N_Id_W_Mont':
            EvalDF[region]['Region'] = 'NW Rockies'
        elif region == 'S_Wyoming':
            EvalDF[region]['Region'] = 'Northern Colorado Rockies'
        elif region == 'SE_Co_Rockies':
            EvalDF[region]['Region'] = 'Sangre de Cristo Mountains'
        elif region == 'Ca_Coast':
            EvalDF[region]['Region'] = 'California Coast Range'
        elif region == 'E_Or':
            EvalDF[region]['Region'] = 'Blue Mountains of Oregon'
        elif region == 'N_Yellowstone':
            EvalDF[region]['Region'] = 'Elkhorn Mountains of Montana '
        elif region == 'S_Cascade':
            EvalDF[region]['Region'] = 'Southern Cascades'
        elif region == 'Wa_Coast':
            EvalDF[region]['Region'] = 'Washington Coast Range '
        elif region == 'Greater_Glacier':
            EvalDF[region]['Region'] = 'Northern Rockies'
        elif region == 'Or_Coast':
            EvalDF[region]['Region'] = 'Oregon Coast Range'

        Model_Results = Model_Results.append(EvalDF[region])
        
    Model_Results['error'] = Model_Results['y_test']-Model_Results['y_pred']
    
    return Model_Results




#Sturm classification of performance
def Sturm_Classified_Performance(Model_Results):
    Maritime_Region = ['Southern Sierra Nevada High','Southern Sierra Nevada Low', 'Northern Sierra Nevada','Southern Cascades',
                      'Northern Cascades', 'California Coast Range', 'Washington Coast Range ', 
                      'Oregon Coast Range']

    Prairie_Region  =  ['Elkhorn Mountains of Montana ','SW Montana', 'Great Basin', 'SW Utah', 'Sawtooth', 'SW Desert']

    Alpine_Region =['Blue Mountains of Oregon', 'Northern Wasatch', 'NW Rockies', 'Greater Yellowstone', 'Upper Colorado Rockies','Northern Colorado Rockies', 'San Juan Mountains',
                         'Northern Rockies', 'Sangre de Cristo Mountains']

    Snow_Class = {'Maritime':Maritime_Region, 
                  'Alpine':Alpine_Region, 
                  # 'Transitional':Transitional_Region, 
                  'Prairie':Prairie_Region}

    for snow in Snow_Class.keys():
        regions = Snow_Class[snow]
        Class = Model_Results[Model_Results['Region'].isin(regions)]
        y_test = Class['y_test']
        y_pred = Class['y_pred']
      #Run model evaluate function
        r2 = sklearn.metrics.r2_score(y_test, y_pred)
        rmse = sklearn.metrics.mean_squared_error(y_test, y_pred, squared = False)
        PBias = he.evaluator(he.pbias, y_pred, y_test)

        print(snow, ' RMSE: ', rmse, ' R2: ', r2, 'pbias:', PBias)
    return Maritime_Region, Prairie_Region, Alpine_Region, Snow_Class




# Figure 3, predicted vs observed for Southern Sierra Nevada, Upper Colorado Rockiesk All regions (subsetted into maritime ,apine, prarie)
def Slurm_Class_parity(Model_Results, Maritime_Region, Prairie_Region, Alpine_Region):

    Model_Results1 = Model_Results[Model_Results['Region'].isin(Maritime_Region)]
    Model_Results2 = Model_Results[Model_Results['Region'].isin(Prairie_Region)]
    Model_Results3 = Model_Results[Model_Results['Region'].isin(Alpine_Region)]

    # fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(3,9))
    font= 10
    tittle_font = font*1.2

    fig = plt.figure(figsize=(5.5,5.5))

    gs=GridSpec(2,2)
    ax1 = fig.add_subplot(gs[:,1])
    ax2 = fig.add_subplot(gs[0,0])
    ax3 = fig.add_subplot(gs[1,0])

    plt.subplots_adjust(hspace=0.2, wspace=0.25)

    #all grouping
    groups_maritime = Model_Results1.groupby('Region')
    for name, group in groups_maritime:
        ax1.plot( group['y_test'],group['y_pred'], marker = 'o', linestyle = ' ', markersize = 2, color='royalblue', label = name, alpha =.2)
    groups_alpine = Model_Results3.groupby('Region')
    for name, group in groups_alpine:
        ax1.plot( group['y_test'],group['y_pred'], marker = 'o', linestyle = ' ', markersize = 2, color='forestgreen', label = name, alpha =.4)
    groups_prairie = Model_Results2.groupby('Region')
    for name, group in groups_prairie:
        ax1.plot( group['y_test'],group['y_pred'], marker = 'o', linestyle = ' ', markersize = 2, color='gold', label = name, alpha =.2)  

    # groups = Model_Results1.groupby('Region')
    # for name, group in groups:
    #     ax1.plot( group['y_test'],group['y_pred'], marker = 'o', linestyle = ' ', markersize = 1, color='grey', label = name)

    ax1.legend(['Maritime', 'Alpine','Prairie'], markerscale=2, handletextpad=0.1, frameon=False)
    leg1=ax1.get_legend()
    for lh in leg1.legendHandles: 
        lh.set_alpha(1)
    leg1.legendHandles[0].set_color('royalblue')
    leg1.legendHandles[1].set_color('forestgreen')
    leg1.legendHandles[2].set_color('gold')
    ax1.plot([0,Model_Results['y_test'].max()], [0,Model_Results['y_test'].max()], color = 'red', linestyle = '--')
    #ax1.set_xlabel('Observed SWE (cm)')
    # ax1.set_ylabel('Predicted SWE (cm)')
    ax1.set_title('All Regions')
    ax1.set_xlim(0,300)
    ax1.set_ylim(0,300)
    ax1.tick_params(axis='y', which='major', pad=1)

    #Sierra Nevada grouping
    groups = Model_Results.loc[(Model_Results["Region"]=="Southern Sierra Nevada High") | (Model_Results["Region"]=="Southern Sierra Nevada Low")].groupby('Region')
    for name, group in groups:
        ax2.plot( group['y_test'],group['y_pred'], marker = 'o', linestyle = ' ', markersize = 2, color='grey', label = name, alpha = .4)

    # ax2.legend(title ='Snow Classification: Prairie', fontsize=font, title_fontsize=tittle_font, ncol = 1, bbox_to_anchor=(1, 1), markerscale = 2)
    ax2.plot([0,Model_Results['y_test'].max()], [0,Model_Results['y_test'].max()], color = 'red', linestyle = '--')
    #ax2.set_xlabel('Observed SWE (cm)')
    # ax2.set_ylabel('Predicted SWE (cm)')
    ax2.set_title('Southern Sierra Nevada')
    ax2.set_xlim(0,300)
    ax2.set_ylim(0,300)
    ax2.xaxis.set_ticklabels([])
    tick_spacing = 100
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax2.tick_params(axis='y', which='major', pad=1)

    #Colorado Rockies Grouping
    groups = Model_Results.loc[(Model_Results["Region"]=="Upper Colorado Rockies")].groupby('Region')
    for name, group in groups:
        ax3.plot( group['y_test'],group['y_pred'], marker = 'o', linestyle = ' ', markersize = 2, color='grey', label = name,alpha = .4)

    # ax3.legend(title ='Snow Classification: Alpine', fontsize=font, title_fontsize=tittle_font, ncol = 1, bbox_to_anchor=(1., 1.), markerscale = 2)
    ax3.plot([0,Model_Results['y_test'].max()], [0,Model_Results['y_test'].max()], color = 'red', linestyle = '--')
    ax3.set_xlabel('Observed SWE (cm)')
    ax3.set_ylabel('Predicted SWE (cm)', labelpad=0)
    ax3.set_title('Upper Colorado Rockies')
    ax3.set_xlim(0,300)
    ax3.set_ylim(0,300)
    ax3.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax3.tick_params(axis='y', which='major', pad=1)

    #save figure
    plt.savefig('./Predictions/Hold_Out_Year/Paper_Figures/Parity_Plot_All4.png', dpi =600, bbox_inches='tight')
    plt.savefig('./Predictions/Hold_Out_Year/Paper_Figures/Parity_Plot_All4.pdf', dpi =600, bbox_inches='tight')
    
    
#Evaluate by Elevation cleaned up
def EvalPlots3(Model_Results, Maritime_Region, Prairie_Region, Alpine_Region, x, y, xlabel, ylabel, plotname, mark_size):
    # fig, (ax1, ax2,ax3) = plt.subplots(2, 1, 2, figsize=(3,9))
    
    Model_Results1 = Model_Results[Model_Results['Region'].isin(Maritime_Region)]
    Model_Results2 = Model_Results[Model_Results['Region'].isin(Prairie_Region)]
    Model_Results3 = Model_Results[Model_Results['Region'].isin(Alpine_Region)]
    
    font= 10
    tittle_font = font*1.2
    
    fig = plt.figure(figsize=(5,6))
    
    gs=GridSpec(2,2)
    ax1 = fig.add_subplot(gs[1,:])
    ax2 = fig.add_subplot(gs[0,0])
    ax3 = fig.add_subplot(gs[0,1])

    plt.subplots_adjust(hspace=0.3, wspace=0.1)

    #all grouping
    groups_maritime = Model_Results1.groupby('Region')
    for name, group in groups_maritime:
        ax1.plot( group[x],group[y], marker = 'o', linestyle = ' ', markersize = mark_size, color='royalblue', label = name, alpha =.2)
    groups_alpine = Model_Results3.groupby('Region')
    for name, group in groups_alpine:
        ax1.plot( group[x],group[y], marker = 'o', linestyle = ' ', markersize = mark_size, color='forestgreen', label = name, alpha =.4)
    groups_prairie = Model_Results2.groupby('Region')
    for name, group in groups_prairie:
        ax1.plot( group[x],group[y], marker = 'o', linestyle = ' ', markersize = mark_size, color='gold', label = name, alpha =.2)  
    
    xmin = min(Model_Results[x])
    xmax = max(Model_Results[x])
    ax1.legend(['Maritime', 'Alpine','Prairie'], markerscale=2, handletextpad=0.1, frameon=False)
    leg1=ax1.get_legend()
    for lh in leg1.legendHandles: 
        lh.set_alpha(1)
    leg1.legendHandles[0].set_color('royalblue')
    leg1.legendHandles[1].set_color('forestgreen')
    leg1.legendHandles[2].set_color('gold')
    ax1.set_title('All Regions')
    ax1.hlines(y=0,xmin = xmin, xmax=xmax, color = 'black', linestyle = '--')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_ylim(-150,150)
    ax1.set_ylabel(ylabel, labelpad=-10)

    #Sierra Nevada grouping
    groups = Model_Results.loc[(Model_Results["Region"]=="Southern Sierra Nevada High") | (Model_Results["Region"]=="Southern Sierra Nevada Low")].groupby('Region')
    for name, group in groups:
        ax2.plot( group[x],group[y], marker = 'o', linestyle = ' ', markersize = mark_size, color='grey', label = name, alpha = .4)
    xmin = min(Model_Results[x])
    xmax = max(Model_Results[x])
    # ax2.legend(title ='Snow Classification: Prairie', fontsize=font, title_fontsize=tittle_font, ncol = 1, bbox_to_anchor=(1, 1), markerscale = 2)
    ax2.set_title('Southern Sierra Nevada')
    ax2.hlines(y=0,xmin = xmin, xmax=xmax,  color = 'black', linestyle = '--')
    #ax2.set_xlabel('Observed SWE (in)')
    # ax2.set_ylabel(ylabel, labelpad=-10)
    ax2.set_ylim(-150,150)



    #Colorado Rockies Grouping
    groups = Model_Results.loc[(Model_Results["Region"]=="Upper Colorado Rockies")].groupby('Region')
    for name, group in groups:
        ax3.plot( group[x],group[y], marker = 'o', linestyle = ' ', markersize = mark_size, color='grey', label = name, alpha = .4)
    xmin = min(Model_Results[x])
    xmax = max(Model_Results[x])
    # ax3.legend(title ='Snow Classification: Alpine', fontsize=font, title_fontsize=tittle_font, ncol = 1, bbox_to_anchor=(1., 1.), markerscale = 2)
    ax3.set_title('Upper Colorado Rockies') 
    ax3.hlines(y=0,xmin = xmin, xmax=xmax,  color = 'black', linestyle = '--')
    ax3.yaxis.set_ticklabels([])
    # ax3.set_xlabel(xlabel)
    # ax3.set_ylabel(ylabel)
    ax3.set_ylim(-150,150)



    # save figure
    plt.savefig(f"./Predictions/Hold_Out_Year/Paper_Figures/{plotname}3.png", dpi =600, bbox_inches='tight')
    plt.savefig(f"./Predictions/Hold_Out_Year/Paper_Figures/{plotname}3.pdf", dpi =600, bbox_inches='tight')
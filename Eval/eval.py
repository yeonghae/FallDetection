# TODO: train.py와 호환성, template.xlsx 없이 작동시키기
## 프레임 길이가 안맞음 노멀 9영상에서
import pandas as pd
import glob
import natsort
import os

NORMAL_LABEL = 0
FALLDOWN_LABEL = 1

template_cols = {'Video':[], 'TotalFrame':[], 'ValidFrame':[], 'FFrame':[], 'NFrame':[], 'Ratio':[], 'Model':[], 'TP':[], 'TN':[], 'FP':[], 'FN':[],
            'Accuracy':[], 'Recall':[], 'Precision':[], 'Specificity':[], 'FPR':[], 'F1-Score':[]}

ret_df = pd.DataFrame(template_cols)

# t_ret_list = glob.glob('_ret/*.xlsx')
t_ret_list = glob.glob('_ret/__Fall200_Normal200_EP300_BS256_LS4.xlsx')
t_ret_list = natsort.natsorted(t_ret_list)

###임시
t_ret_list = ['Eval/__F300_N200_LS4_NF20_EP50_BS256_UseShotFrameAllFallLabel_UseEvalNF17.xlsx']

m_list = []
# 평가할 데이터 로드  # 멀티 스레딩 처리 해야됨 
t_mdf_list = []
for i, m_name in enumerate(t_ret_list):
    _df = pd.read_excel(m_name)
    col_list = _df.columns.to_list()
    col_list[0] = col_list[0].split('\\')[-1]
    col_list[0] = col_list[0].split('_Prediction')[0]

    _df.columns = col_list
    m_list.append(col_list[0])
    t_mdf_list.append(_df.iloc[:, 0])
    print(i)

    if i==2:
        break



# 엑셀 파일을 DataFrame으로 로드
os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_excel('template_result.xlsx')
v_list = df.drop_duplicates(subset='Video')['Video'].to_list()
v_list = natsort.natsorted(v_list)

# 모델별 결과와 템플릿 병합
m_df = pd.concat(t_mdf_list, axis=1)
df = pd.concat([df, m_df], axis=1)

for v_name in v_list:
    v_df = df[df['Video'] == v_name].copy()
    total_f = len(v_df)
#길이가 안맞는 버그 
    for m_name in m_list:
        predict_name_col = m_name
        v_df = v_df[v_df[predict_name_col]!=-1]

        len_row = len(v_df)                
        GT2_col = v_df['GroundTruth'].copy()
        l2_col = v_df[predict_name_col].copy()

        GT2_col.loc[GT2_col == 2] = 0
        l2_col[l2_col == 2] = 0

        len_ture = len(v_df[(l2_col == FALLDOWN_LABEL)])
        len_false = len(v_df[(l2_col == NORMAL_LABEL)])

        #실제로 낙상이고, 낙상으로 예측
        true_positive = v_df[(GT2_col == l2_col) & (l2_col == FALLDOWN_LABEL)]
        #실제로 일반이고, 일반으로 예측
        true_nagative = v_df[(GT2_col == l2_col) & (l2_col == NORMAL_LABEL)]
        #실제로 일반이고, 낙상으로 예측
        false_positive = v_df[(GT2_col != l2_col) & (l2_col == FALLDOWN_LABEL)]
        #실제로 낙상이고, 일반으로 예측
        false_nagative = v_df[(GT2_col != l2_col) & (l2_col == NORMAL_LABEL)]


        if(len_row != len_ture+len_false):
            print(v_name, "row 갯수 검증 에러")
            exit()

        total_cm = len(true_positive)+len(true_nagative)+len(false_positive)+len(false_nagative)
        if(len_row != total_cm):
            print("cm과 row 갯수 검증 에러")
            exit()

        TP = len(true_positive)
        TN = len(true_nagative)
        FP = len(false_positive)
        FN = len(false_nagative)


        accuracy = (TP+TN)/(TP+TN+FP+FN)
        # 모델이 Positive라 분류한 것 중 실제 값이 Positive인 비율
        try:
           precision = TP/(TP+FP)
        except:
            precision = 0        
        # 실제 값이 Positive 인 것 중 모델이 Positive라 분류한 비율
        try:
            recall = TP/(TP+FN)
        except:
            recall = 0    
        # F1 Socre Precision과 Recall의 조화평균
        
        try:
            f1_socre = (2*precision*recall) / (precision+recall)
        except:
            f1_socre = 0

        if(TN+FP == 0):
            specificity = 0
        else:
            specificity = TN/(TN+FP)
    
        if(FP+TN == 0):
            fpr = 0
        else:
            fpr = FP/(FP+TN)
        # tpr = TP/(TP+FN)
        # roc_auc = #'ROC_AUC':

        new_row = pd.DataFrame({'Video':[v_name], 'TotalFrame':[total_f], 'ValidFrame':[len_row], 'Model':[m_name], 'TP':[TP], 'TN':[TN], 'FP':[FP], 'FN':[FN],
                   'Accuracy':[accuracy], 'Recall':[recall], 'Precision':[precision], 'Specificity':[specificity], 'FPR':[fpr], 'F1-Score':[f1_socre]})
        ret_df = pd.concat([ret_df, new_row], ignore_index=True)
ret_df.to_csv('ret_per_video.csv')


ret_m_df = pd.DataFrame(template_cols)
ret_df = ret_df.drop('Video', axis=1)
for m_name in m_list:
    m_df = ret_df[ret_df['Model'] == m_name].copy()
    m_df = m_df.drop('Model', axis=1)
    m_df = m_df.sum(axis=0)
    m_df['Model'] = m_name

    print(m_df)
    TP = m_df['TP']
    TN = m_df['TN']
    FP = m_df['FP']
    FN = m_df['FN']

    accuracy = (TP+TN)/(TP+TN+FP+FN)

    # 모델이 Positive라 분류한 것 중 실제 값이 Positive인 비율
    try:
        precision = TP/(TP+FP)
    except:
        precision = 0        
    # 실제 값이 Positive 인 것 중 모델이 Positive라 분류한 비율
    try:
        recall = TP/(TP+FN)
    except:
        recall = 0    
    # F1 Socre Precision과 Recall의 조화평균
    
    try:
        f1_socre = (2*precision*recall) / (precision+recall)
    except:
        f1_socre = 0

    specificity = TN/(TN+FP)
    fpr = FP/(FP+TN)

    new_row = pd.DataFrame({'Model':[m_name], 'TotalFrame':[m_df['TotalFrame']], 'ValidFrame':[m_df['ValidFrame']], 'TP':[m_df['TP']], 'TN':[m_df['TN']], 'FP':[m_df['FP']], 'FN':[m_df['FN']],
            'Accuracy':[accuracy], 'Recall':[recall], 'Precision':[precision], 'Specificity':[specificity], 'FPR':[fpr], 'F1-Score':[f1_socre]})

    ret_m_df = pd.concat([ret_m_df, new_row], ignore_index=True)

ret_m_df.to_csv('ret_per_model.csv')

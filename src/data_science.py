def validation_curves(model,X_train,y_train,X_cv,y_cv,scoring,path_,cv=5,original_or_positive='original',parameters_range={},save=False,
                    color_line=None):
    from sklearn.model_selection import validation_curve
    import json 
    import plotly
    import os
    import numpy as np
    import pandas as pd
    import plotly.express as px

    if os.path.exists(os.path.join(path_,str(model.__class__.__name__))):
        path_ = os.path.join(path_,str(model.__class__.__name__))
    else:
        os.mkdir(os.path.join(path_,str(model.__class__.__name__)))
        path_= os.path.join(path_,str(model.__class__.__name__))

    """
    Dataframes in numpy form
    parameters_range: a dictionarie with key = str, and range = list
    """

    if 'numpy' in str(type(X_train)):
        X = np.concatenate((X_train,X_cv),axis=0)
        y = np.concatenate((y_train,y_cv),axis=0)       
    else:
        X=pd.concat([X_train,X_cv],axis=0)
        y=pd.concat([y_train,y_cv],axis=0)

    memory_ = 0
    memory_file = []

    if save == True:
        try:       
            conter = np.load(os.path.join(path_,'conter.npy'))
            fig_conter = np.load(os.path.join(path_,'fig_conter.npy'))

        except:
            conter = np.array(1)
            fig_conter = np.array(0)
            np.save('conter',conter)
            with open(os.path.join(path_,'validation_curves.json'),'a',encoding="utf-8") as f:
                f.write(json.dumps([parameters_range])) #json file that contain the curves made
            with open(os.path.join(path_,'validation_curves_report.txt'),'a',encoding="utf-8") as f:
                j = 1
                for i,k in parameters_range.items():
                        f.write(str(i)+ " : "+str(k)+ "-> figure name: "+'validation_curve_'+str(j)+f'\n')
                        j+=1

        if conter != 1: 
            with open(os.path.join(path_,'validation_curves.json'),'r+',encoding="utf-8") as f:
                read_ = f.read()

            data = json.loads(read_).copy()

            for k in data:
                for h,t in k.items():
                    try:
                        if parameters_range[h] == t: # quita las figuras que se repetirian, es decir las ya realizadas
                            memory_ +=1 # cuenta el numero de figuras ya realizadas
                            memory_file.append(h + ' : ' + str(t))
                            del parameters_range[h]
                    except:
                        pass

            data.append(parameters_range)

            with open(os.path.join(path_,'validation_curves.json'),'r+',encoding="utf-8") as f:
                f.write(json.dumps(data))

            with open(os.path.join(path_,'validation_curves_report.txt'),'a',encoding="utf-8") as f:
                j = 1
                for i,k in parameters_range.items():
                        f.write(str(i) + " : " + str(k) + "-> figure name: " + 'validation_curve_'+str(fig_conter+j)+f'\n')
                        j+=1

    else:
        try: 
            fig_conter = np.load(os.path.join(path_,'fig_conter.npy'))
            with open(os.path.join(path_,'validation_curves.json'),'r+',encoding="utf-8") as f:
                read_ = f.read()

            data = json.loads(read_).copy()

            for k in data:
                for h,t in k.items():
                    try:
                        if parameters_range[h] == t: # quita las figuras que se repetirian, es decir las ya realizadas
                            memory_ +=1 # cuenta el numero de figuras ya realizadas
                            memory_file.append(h + ' : ' + str(t))
                            del parameters_range[h]
                    except:
                        pass
        except:
            pass 

    fig = []

    if  memory_ != 0:

        for i in range(1,memory_+1):
                with open(os.path.join(path_,'validation_curves_report.txt'),'r',encoding="utf-8") as f:
                    l = f.readlines()
                    for i in range(0,len(l)):
                        a,b = l[i].split('-> ')
                        for i in memory_file:                          
                            if i == a:
                                curve = b[13:-1]
                                fig.append(plotly.io.read_json((os.path.join(path_,curve+'.json'))))

    for i,k in parameters_range.items():
        if save == True:
            fig_conter+=1
            conter+=1
            np.save(os.path.join(path_,'conter'),conter)
            
        train_scores, cv_scores = validation_curve(model, X, y.values.ravel(), param_name=i, param_range=k,cv=cv,scoring=scoring,n_jobs=-1)
        if original_or_positive == 'positive':
            train_scores_mean = np.abs(np.mean(train_scores, axis=1))
            train_scores_std = np.abs(np.std(train_scores, axis=1))
            cv_scores_mean = np.abs(np.mean(cv_scores, axis=1))
            cv_scores_std = np.abs(np.std(cv_scores, axis=1))
        else:
            train_scores_mean = (np.mean(train_scores, axis=1))
            train_scores_std = (np.std(train_scores, axis=1))
            cv_scores_mean = (np.mean(cv_scores, axis=1))
            cv_scores_std = (np.std(cv_scores, axis=1))

        mae_train_cv_poly_graph = px.line(y=train_scores_mean, x=k, 
            title = i.replace('_',' ').capitalize() +' Validation Curve',
            labels={'x':i.replace('_',' ').capitalize(),
                'y':scoring[4:].replace('_',' ').capitalize()},markers=True) 
        mae_train_cv_poly_graph.update_traces(showlegend=True,name='Train',line_color=color_line[0],hovertemplate=None)
        mae_train_cv_poly_graph.add_scatter(y=cv_scores_mean,x=k,name='Cv',line_color=color_line[1])
        mae_train_cv_poly_graph.update_layout(hovermode="x")

        if save == True:
            with open(os.path.join(path_,'validation_curve_'+str(fig_conter)+'.json'),'w',encoding="utf-8") as f:
                f.write(plotly.io.to_json(mae_train_cv_poly_graph))
            
            np.save(os.path.join(path_,'fig_conter'),fig_conter)

        fig.append(mae_train_cv_poly_graph)

    return fig # Figures list

def table(df,bgcolor="#fff",textcolor="#fff",bgheader='#fff',columns_description=[None]):
    from dash import dash_table
    table = dash_table.DataTable( 
        columns = [{"name": i, "id": i, "deletable": False, "selectable": False, "hideable": False} for i in df.columns], 
        data = df.round(decimals=4).to_dict('records'),
        editable=True,              # allow editing of data inside all cells
        filter_action="native",     # allow filtering of data by user ('native') or not ('none')
        sort_action="native",       # enables data to be sorted per-column by user or not ('none')
        sort_mode="single",         # sort across 'multi' or 'single' columns
        column_selectable="multi",  # allow users to select 'multi' or 'single' columns
        row_selectable=False,     # allow users to select 'multi' or 'single' rows
        row_deletable=False,         # choose if user can delete a row (True) or not (False)
        selected_columns=[],       # ids of columns that user selects
        selected_rows=[],           # indices of rows that user selects
        page_action="none",       # all data is passed to the table up-front or not ('none')
        page_current=0,             # page number that user is on
        page_size=7,                # number of rows visible per page
        style_cell={                # ensure adequate header width when text is shorter than cell's text
            'minWidth': '220px', 'width': '220px', 'maxWidth': '223px', "whiteSpace": "pre-line"
        }, # "whiteSpace": "pre-line": permite dividir la información de las celdas en más de una columna
        style_data={                # overflow cells' content into multiple lines
            'whiteSpace': 'normal',
            'height': 'auto',
            'color': textcolor,
            'backgroundColor': bgcolor
        },
        style_table={'overflow':'scroll','height':'20rem'},
        style_header={
            'backgroundColor': bgheader,
             'color': textcolor,
          },
        style_filter={
            'backgroundColor': bgheader,
             'color': textcolor,
          },
        fixed_rows={'headers': True, 'data': 0},
        css=[{"selector": ".show-hide", "rule": "display: none; "},
        {"selector":".dash-table-tooltip","rule":"background-color:{0}; color:{1};".format(bgheader,textcolor)}],
        tooltip_header={i: k for i,k in zip(df.columns,columns_description)},
        tooltip_delay=0,
        tooltip_duration=None
        #style_cell_conditional=[    # align text columns to left. By default they are aligned to right
        #    {
        #        'if': {'column_id': c},
        #        'textAlign': 'left'
        #    } for c in X[2:].columns
        #],
        )
    return table

def load_plot_json(path,name):
    import plotly
    import json 
    import os
    with open(os.path.join(path,name+'.json'),'r+',encoding="utf-8") as f:
        read_ = f.read()
    figure = plotly.io.from_json(read_)
    return figure
import pandas as pd
import streamlit as st
import plotly.express as px
from src.ai_engine import *

st.set_page_config(page_title='AI Expense Behavior Analyzer', page_icon='💰', layout='wide')
st.title('💰 AI Expense Behavior Analyzer')
st.caption('Enter your own financial data and analyze spending with AI/ML.')

if 'expenses' not in st.session_state:
    st.session_state.expenses = prepare_df(pd.read_csv('data/sample_expenses.csv'))
if 'salary' not in st.session_state: st.session_state.salary = 50000.0
if 'manual_history' not in st.session_state: st.session_state.manual_history = []

tabs=st.tabs(['📊 Dashboard','✍️ Manual Data Entry','➕ Add Expense','📁 Upload Data','🤖 AI Analysis','🔮 Forecast & Budget','💬 AI Assistant'])

with tabs[0]:
    df=st.session_state.expenses; salary=st.session_state.salary; total=df.amount.sum(); balance=salary-total
    a,b,c,d=st.columns(4)
    a.metric('Total Expenses',f'₹{total:,.0f}'); b.metric('Monthly Salary',f'₹{salary:,.0f}')
    c.metric('Estimated Balance',f'₹{balance:,.0f}'); d.metric('Balance %',f'{balance/salary*100:.1f}%' if salary else '0%')
    x=df.groupby('category',as_index=False).amount.sum()
    st.plotly_chart(px.bar(x,x='category',y='amount',title='Spending by Category'),width='stretch')
    m=df.groupby('month',as_index=False).amount.sum()
    st.plotly_chart(px.line(m,x='month',y='amount',markers=True,title='Monthly Spending Trend'),width='stretch')
    st.dataframe(df.sort_values('date',ascending=False).head(15),width='stretch',hide_index=True)

with tabs[1]:
    st.header('✍️ Manual Financial Data Entry')
    st.write('Enter your monthly salary and spending by category. No bank file is required.')
    salary=st.number_input('Monthly Salary / Income (₹)',min_value=0.0,value=float(st.session_state.salary),step=1000.0)
    month=st.date_input('Month',value=pd.Timestamp.today().date())
    st.subheader('Category-wise Monthly Spending')
    vals={}; cols=st.columns(3); existing=st.session_state.expenses.groupby('category').amount.sum().to_dict()
    for i,cat in enumerate(CATEGORIES):
        with cols[i%3]: vals[cat]=st.number_input(f'{cat} (₹)',min_value=0.0,value=float(existing.get(cat,0)),step=100.0,key='manual_'+cat)
    if st.button('💾 Save Manual Financial Data',type='primary'):
        st.session_state.salary=float(salary)
        month_str=str(pd.Timestamp(month).to_period('M'))
        rows=[{'date':f'{month_str}-01','description':f'Manual {cat} spending','amount':amt,'category':cat} for cat,amt in vals.items() if amt>0]
        if rows: st.session_state.expenses=prepare_df(pd.concat([st.session_state.expenses,prepare_df(pd.DataFrame(rows))],ignore_index=True))
        st.session_state.manual_history.append({'month':month_str,'salary':salary,'total_spending':sum(vals.values())})
        st.success('Your salary and category-wise spending have been saved.'); st.rerun()
    if st.session_state.manual_history: st.dataframe(pd.DataFrame(st.session_state.manual_history),width='stretch',hide_index=True)
    st.download_button('⬇️ Download Current Data',st.session_state.expenses.to_csv(index=False),'my_expense_data.csv','text/csv')

with tabs[2]:
    st.header('➕ Add Individual Expense')
    with st.form('expense'):
        date=st.date_input('Date',value=pd.Timestamp.today().date()); desc=st.text_input('Description'); amt=st.number_input('Amount (₹)',min_value=0.0,step=100.0); cat=st.selectbox('Category',CATEGORIES+['AI Auto-Detect']); ok=st.form_submit_button('Add Expense')
    if ok:
        if not desc.strip() or amt<=0: st.error('Enter a description and amount.')
        else:
            cat=predict_categories([desc])[0] if cat=='AI Auto-Detect' else cat
            st.session_state.expenses=prepare_df(pd.concat([st.session_state.expenses,prepare_df(pd.DataFrame([{'date':date,'description':desc,'amount':amt,'category':cat}]))],ignore_index=True))
            st.success(f'Added ₹{amt:,.2f} under {cat}.'); st.rerun()

with tabs[3]:
    st.header('📁 Upload CSV / Excel')
    f=st.file_uploader('Upload expense data',type=['csv','xlsx'])
    if f:
        x=prepare_df(pd.read_csv(f) if f.name.lower().endswith('.csv') else pd.read_excel(f))
        st.dataframe(x.head(20),width='stretch',hide_index=True)
        if st.button('Use Uploaded Data'): st.session_state.expenses=x; st.rerun()

with tabs[4]:
    st.header('🤖 AI Spending Analysis')
    for i in generate_insights(st.session_state.expenses,st.session_state.salary): st.info('💡 '+i)
    st.subheader('Spending Behavior Profile'); st.dataframe(behavior_profile(st.session_state.expenses),width='stretch',hide_index=True)
    st.subheader('🚨 Anomalies'); an=detect_anomalies(st.session_state.expenses); st.dataframe(an[an.anomaly],width='stretch',hide_index=True)
    st.subheader('🔁 Recurring'); st.dataframe(recurring_merchants(st.session_state.expenses),width='stretch',hide_index=True)
    st.subheader('♻️ Duplicate Candidates'); st.dataframe(find_duplicate_candidates(st.session_state.expenses),width='stretch',hide_index=True)

with tabs[5]:
    st.header('🔮 Forecast & Smart Budget'); st.metric('Predicted Next-Month Expenses',f'₹{forecast_next_month(st.session_state.expenses):,.0f}')
    b=smart_budget(st.session_state.expenses,st.session_state.salary); st.dataframe(b,width='stretch',hide_index=True)
    if not b.empty: st.plotly_chart(px.bar(b,x='category',y='recommended_budget',title='Smart Budget Recommendation'),width='stretch')

with tabs[6]:
    st.header('💬 AI Financial Assistant'); q=st.text_input('Ask about your spending',placeholder='How much did I spend on food?')
    if q:
        df=st.session_state.expenses; q=q.lower()
        if 'salary' in q or 'income' in q: st.success(f'Monthly salary: ₹{st.session_state.salary:,.2f}')
        elif 'balance' in q or 'left' in q: st.success(f'Estimated balance: ₹{st.session_state.salary-df.amount.sum():,.2f}')
        elif 'food' in q: st.success(f'Food spending: ₹{df.loc[df.category.str.lower()=="food", "amount"].sum():,.2f}')
        elif 'highest' in q or 'most' in q:
            x=df.groupby('category').amount.sum(); st.success(f'Highest category: {x.idxmax()} — ₹{x.max():,.2f}')
        else: st.success(f'Total recorded expenses: ₹{df.amount.sum():,.2f}')

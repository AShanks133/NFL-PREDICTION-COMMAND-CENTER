import datetime as dt
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title='NFL Prediction Command Center',page_icon='🏈',layout='centered',initial_sidebar_state='collapsed')
st.markdown('''<style>
@media (max-width: 650px) {
.block-container {padding-top: 1.1rem; padding-left: .75rem; padding-right: .75rem;}
h1 {font-size: 1.65rem !important;}
[data-testid="stMetricValue"] {font-size: 1.45rem;}
button {min-height: 2.5rem;}
}
</style>''',unsafe_allow_html=True)
st.title('🏈 NFL Prediction Command Center')
st.caption('2026 NFL • Weekly slate + matchup lab • Monte Carlo projections • Not betting advice')
API='https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard'
SEASON=2026

@st.cache_data(ttl=1800,show_spinner=False)
def fetch_season():
    rows=[]; upcoming=[]; errors=[]
    with requests.Session() as session:
        for week in range(1,19):
            try:
                r=session.get(API,params={'dates':SEASON,'seasontype':2,'week':week,'limit':100},timeout=12)
                r.raise_for_status(); events=r.json().get('events',[])
            except Exception as e:
                errors.append(f'Week {week}: {type(e).__name__}'); continue
            for ev in events:
                comp=(ev.get('competitions') or [{}])[0]; players=comp.get('competitors',[])
                if len(players)!=2:continue
                home=next((p for p in players if p.get('homeAway')=='home'),None)
                away=next((p for p in players if p.get('homeAway')=='away'),None)
                if not home or not away:continue
                base={'id':ev.get('id'),'week':week,'date':ev.get('date',''),'home':home.get('team',{}).get('displayName'),'away':away.get('team',{}).get('displayName')}
                if not base['home'] or not base['away']:continue
                if ev.get('status',{}).get('type',{}).get('completed'):
                    try:rows.append({**base,'home_pts':int(home['score']),'away_pts':int(away['score'])})
                    except (KeyError,ValueError,TypeError):pass
                elif ev.get('status',{}).get('type',{}).get('state')=='pre':upcoming.append(base)
    return pd.DataFrame(rows).drop_duplicates('id') if rows else pd.DataFrame(),pd.DataFrame(upcoming).drop_duplicates('id') if upcoming else pd.DataFrame(),errors

with st.sidebar:
    st.header('Simulation controls')
    n=st.select_slider('Runs per matchup',[1000,5000,10000,25000,50000,100000],value=25000)
    recent=st.slider('Weight on last 3 games',0.0,0.65,0.25,0.05)
    home_edge=st.slider('Home-field points',0.0,4.0,1.5,0.25)
    volatility=st.slider('Scoring volatility',6.,18.,12.,0.5)
    shrink=st.slider('Regression to league average',0.,0.6,0.2,0.05)
    seed=st.number_input('Random seed',value=2026,min_value=0,step=1)
    st.button('🔄 Refresh feed',on_click=st.cache_data.clear)
with st.spinner('Fetching 2026 NFL schedule and completed games…'):
    games,upcoming,errors=fetch_season()
if games.empty:
    st.error('No completed games available. ESPN feed may be unavailable; check connection and refresh.');st.stop()
long=pd.concat([games.rename(columns={'home':'team','away':'opponent','home_pts':'scored','away_pts':'allowed'}).assign(venue='home'),games.rename(columns={'away':'team','home':'opponent','away_pts':'scored','home_pts':'allowed'}).assign(venue='away')],ignore_index=True).sort_values(['date','week'])
league=float(long.scored.mean())
teams=sorted(long.team.unique())

def ratings(history):
    L=pd.concat([history.rename(columns={'home':'team','away':'opponent','home_pts':'scored','away_pts':'allowed'}),history.rename(columns={'away':'team','home':'opponent','away_pts':'scored','home_pts':'allowed'})],ignore_index=True).sort_values(['date','week'])
    avg=float(L.scored.mean()); data={}
    for team,x in L.groupby('team'):
        x=x.sort_values(['date','week']); tail=x.tail(3)
        scored=(1-recent)*x.scored.mean()+recent*tail.scored.mean()
        allowed=(1-recent)*x.allowed.mean()+recent*tail.allowed.mean()
        data[team]=(scored,allowed,len(x))
    return data,avg
rating,avg=ratings(games)

def simulate(away,home,neutral=False,count=n,random_seed=seed,rt=None,league_avg=None):
    rt=rating if rt is None else rt; league_avg=avg if league_avg is None else league_avg
    ap,aa,an=rt[away]; hp,ha,hn=rt[home]
    mu_a=(1-shrink)*(ap*ha/max(league_avg,1))+shrink*league_avg
    mu_h=(1-shrink)*(hp*aa/max(league_avg,1))+shrink*league_avg+(0 if neutral else home_edge)
    rng=np.random.default_rng(int(random_seed)); shared=rng.normal(0,volatility*.35,count)
    a=np.maximum(0,np.rint(rng.normal(mu_a,volatility*.94,count)+shared)).astype(int)
    h=np.maximum(0,np.rint(rng.normal(mu_h,volatility*.94,count)+shared)).astype(int)
    tie=h==a; coin=rng.random(count)<.5; h[tie & coin]+=3; a[tie & ~coin]+=3
    return a,h,{'away_expected':mu_a,'home_expected':mu_h,'away_games':an,'home_games':hn}

def stats(a,h,spread=None,total_line=None):
    m=h-a;t=h+a
    out={'Home win %':round(100*np.mean(m>0),1),'Away win %':round(100*np.mean(m<0),1),'Projected home':round(float(h.mean()),1),'Projected away':round(float(a.mean()),1),'Projected margin':round(float(m.mean()),1),'Projected total':round(float(t.mean()),1),'Home win 80% margin low':float(np.quantile(m,.1)),'Home win 80% margin high':float(np.quantile(m,.9))}
    if spread is not None:out.update({'Home cover %':round(100*np.mean(m+spread>0),1),'Spread push %':round(100*np.mean(m+spread==0),1)})
    if total_line is not None:out.update({'Over %':round(100*np.mean(t>total_line),1),'Under %':round(100*np.mean(t<total_line),1),'Total push %':round(100*np.mean(t==total_line),1)})
    return out

st.success(f'{len(games)} completed games loaded • Last completed: {games.date.max()[:10]} • League average: {avg:.1f} points/team/game')
if errors:st.warning(f'Feed unavailable for {len(errors)} of 18 weeks; results may be incomplete.')
tab_slate,tab_match,tab_backtest,tab_data=st.tabs(['📅 Weekly predictions','🎯 Matchup simulator','🧪 Historical backtest','📋 Data & methodology'])
with tab_slate:
    if upcoming.empty:st.warning('No upcoming regular-season games returned by feed.')
    else:
        weeks=sorted(upcoming.week.unique().tolist()); today=dt.datetime.now(dt.timezone.utc)
        future=upcoming[pd.to_datetime(upcoming.date,utc=True)>=today]
        default_week=int(future.week.min()) if not future.empty else int(weeks[-1])
        wk=st.selectbox('Select NFL week',weeks,index=weeks.index(default_week))
        slate=upcoming[upcoming.week==wk].sort_values('date')
        rows=[]
        for i,(_,g) in enumerate(slate.iterrows()):
            if g.away not in rating or g.home not in rating:continue
            a,h,_=simulate(g.away,g.home,random_seed=int(seed)+i)
            s=stats(a,h)
            rows.append({'Kickoff (UTC)':g.date[:16].replace('T',' '),'Away':g.away,'Home':g.home,'Favorite':g.home if s['Home win %']>=50 else g.away,'Favorite win %':max(s['Home win %'],s['Away win %']),'Projected score (away–home)':f"{s['Projected away']:.1f} – {s['Projected home']:.1f}",'Projected home margin':s['Projected margin'],'Projected total':s['Projected total'],'Confidence tier':('Higher' if max(s['Home win %'],s['Away win %'])>=70 else 'Moderate' if max(s['Home win %'],s['Away win %'])>=60 else 'Close')})
        if rows:
            frame=pd.DataFrame(rows).sort_values('Favorite win %',ascending=False)
            st.dataframe(frame,use_container_width=True,hide_index=True)
            st.download_button('Download weekly projections CSV',frame.to_csv(index=False),'nfl_week_predictions.csv','text/csv')
            st.caption('Confidence tiers describe simulated win probabilities, not proven predictive accuracy. Odds are not fetched automatically.')
        else:st.info('Insufficient completed-game history for teams in this slate.')
with tab_match:
    c1,c2=st.columns(2)
    with c1:away=st.selectbox('Away team',teams,index=0)
    with c2:home=st.selectbox('Home team',teams,index=1)
    neutral=st.checkbox('Neutral site',False)
    c3,c4=st.columns(2)
    with c3:spread=st.number_input('Home spread (negative = favored)',value=-3.5,step=.5)
    with c4:total_line=st.number_input('Over/under line',value=44.5,step=.5)
    if home==away:st.warning('Select two different teams.')
    else:
        a,h,inputs=simulate(away,home,neutral)
        s=stats(a,h,spread,total_line)
        cols=st.columns(4)
        cols[0].metric('Home win probability',f"{s['Home win %']}%")
        cols[1].metric('Away win probability',f"{s['Away win %']}%")
        cols[2].metric('Projected score (away–home)',f"{s['Projected away']} – {s['Projected home']}")
        cols[3].metric('Projected total',s['Projected total'])
        st.write(f"**Home covers {spread:+g}:** {s['Home cover %']}% • **Push:** {s['Spread push %']}% • **Over {total_line:g}:** {s['Over %']}% • **Under:** {s['Under %']}%")
        st.write(f"**Projected home margin:** {s['Projected margin']:+.1f} • **80% simulated margin interval:** {s['Home win 80% margin low']:+.0f} to {s['Home win 80% margin high']:+.0f}")
        st.subheader('Simulated margin distribution');st.bar_chart(pd.Series(h-a).value_counts().sort_index())
        st.subheader('Simulated total distribution');st.bar_chart(pd.Series(h+a).value_counts().sort_index())
        st.dataframe(pd.DataFrame([inputs]),hide_index=True)
        st.download_button('Download all simulations CSV',pd.DataFrame({'away_score':a,'home_score':h,'home_margin':h-a,'total':h+a}).to_csv(index=False),'nfl_matchup_simulations.csv','text/csv')
with tab_backtest:
    st.write('Walk-forward test: for each historical game, train only on **earlier completed games**. This avoids using future results to predict past games.')
    min_games=st.slider('Minimum prior games per team',1,6,3)
    ordered=games.sort_values(['date','week']).reset_index(drop=True)
    records=[]
    for i,g in ordered.iterrows():
        prior=ordered[ordered.date<g.date]
        if prior.empty:continue
        prior_rt,prior_avg=ratings(prior)
        if g.away not in prior_rt or g.home not in prior_rt:continue
        if min(prior_rt[g.away][2],prior_rt[g.home][2])<min_games:continue
        a,h,_=simulate(g.away,g.home,count=5000,random_seed=int(seed)+i,rt=prior_rt,league_avg=prior_avg)
        p=float(np.mean(h>a)); actual=int(g.home_pts>g.away_pts)
        records.append({'Week':g.week,'Game':f'{g.away} at {g.home}','Home win forecast':p,'Home won':actual,'Predicted home margin':float((h-a).mean()),'Actual home margin':int(g.home_pts-g.away_pts)})
    if not records:st.info('Not enough prior games for a backtest yet.')
    else:
        b=pd.DataFrame(records)
        acc=np.mean((b['Home win forecast']>=.5)==b['Home won'])
        brier=np.mean((b['Home win forecast']-b['Home won'])**2)
        mae=np.mean(np.abs(b['Predicted home margin']-b['Actual home margin']))
        c=st.columns(4);c[0].metric('Games tested',len(b));c[1].metric('Winner accuracy',f'{acc:.1%}');c[2].metric('Brier score (lower better)',f'{brier:.3f}');c[3].metric('Margin MAE',f'{mae:.1f} pts')
        st.dataframe(b,hide_index=True,use_container_width=True)
        st.download_button('Download backtest CSV',b.to_csv(index=False),'nfl_backtest.csv','text/csv')
        st.caption('Backtest reflects the current model settings; these settings were not optimized on a held-out sample. Same-day games are excluded from training.')
with tab_data:
    st.subheader('Current team inputs')
    st.dataframe(pd.DataFrame([{'Team':k,'Weighted scored/game':v[0],'Weighted allowed/game':v[1],'Games':v[2]} for k,v in rating.items()]).sort_values('Team'),hide_index=True,use_container_width=True)
    st.markdown('**Method:** Offensive points/game × opponent defensive points allowed/game ÷ league average, regressed toward league average; recent-form blending; home-field adjustment; correlated normal scoring noise; tie resolution approximation. Every simulation is a sampled *full-game outcome*, not a play-by-play engine.')
    st.warning('Limitations: This model does not yet use EPA/play, drive-level pace, QB injuries, weather, opponent-adjusted strength of schedule, or live sportsbook lines. It does not auto-refresh while a game is underway. Probabilities may be miscalibrated; backtesting and out-of-sample validation are essential. Betting edges cannot be asserted without real market prices and a validated model.')
    st.caption('Data: ESPN public scoreboard endpoint; unofficial feed availability and schema can change. Refresh every 30 minutes or manually. No affiliation with NFL or ESPN.')

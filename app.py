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
@st.cache_data(ttl=21600,show_spinner=False)
def team_id_map():
    """Map display names to ESPN numeric IDs, never use names in roster URLs."""
    mapping={}
    try:
        response=requests.get('https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams',params={'limit':100},timeout=12)
        response.raise_for_status()
        data=response.json()
        sports=data.get('sports',[])
        for sport in sports:
            for league in sport.get('leagues',[]):
                for entry in league.get('teams',[]):
                    t=entry.get('team',entry)
                    for key in ('displayName','name','shortDisplayName'):
                        if t.get(key):mapping[t[key]]=str(t.get('id',''))
    except (requests.RequestException,ValueError,TypeError):
        pass
    return mapping

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

# Player-prop module: ESPN game summaries, with explicit data-coverage checks.
st.divider()
st.header('🏈 Player Prop Lab')
st.caption('Player game logs from ESPN game summaries (when available). These are statistical estimates, not validated betting edges.')
PROP_KEYS = {'Passing yards':('passing','YDS'),'Passing touchdowns':('passing','TD'),'Pass completions':('passing','C/ATT'),'Rushing yards':('rushing','YDS'),'Rushing attempts':('rushing','CAR'),'Receiving yards':('receiving','YDS'),'Receptions':('receiving','REC'),'Targets':('receiving','TGTS')}
SUMMARY_URL='https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary'
@st.cache_data(ttl=3600,show_spinner=False)
def player_game_logs(game_ids):
    result=[]
    for gid in game_ids:
        try:
            response=requests.get(SUMMARY_URL,params={'event':str(gid)},timeout=8)
            response.raise_for_status()
            payload=response.json()
            for team in payload.get('boxscore',{}).get('players',[]):
                team_name=team.get('team',{}).get('displayName','')
                for category in team.get('statistics',[]):
                    cat=category.get('name','').lower()
                    labels=category.get('labels',[])
                    for athlete in category.get('athletes',[]):
                        name=athlete.get('athlete',{}).get('displayName','')
                        values=athlete.get('stats',[])
                        for prop,(wanted,label) in PROP_KEYS.items():
                            if cat!=wanted or label not in labels:continue
                            idx=labels.index(label)
                            if idx>=len(values):continue
                            raw=str(values[idx])
                            if label=='C/ATT':raw=raw.split('/')[0]
                            try:value=float(raw.replace(',',''))
                            except ValueError:continue
                            result.append({'game_id':str(gid),'team':team_name,'player':name,'prop':prop,'value':value})
        except (requests.RequestException,ValueError,KeyError,TypeError):continue
    return pd.DataFrame(result)

with st.expander('📊 Load player statistics and simulations',expanded=True):
    st.info('Player statistics are downloaded on demand. ESPN may limit or change access to historical box scores. Loading many games can take time.')
    weeks_back=st.slider('Recent completed weeks to load',2,10,6,key='prop_weeks')
    if st.button('📥 Load player game logs',key='load_player_logs'):
        recent_games=games[games.week>=max(1,int(games.week.max())-weeks_back+1)].sort_values('date')
        with st.spinner(f'Retrieving up to {len(recent_games)} box scores…'):
            st.session_state['player_logs']=player_game_logs(tuple(recent_games.id.astype(str).tolist()))
    logs=st.session_state.get('player_logs',pd.DataFrame())
    if logs.empty:
        st.warning('No player game logs loaded. Tap the load button. If it still returns no data, the ESPN player-stat feed may be unavailable.')
    else:
        st.success(f'{logs.game_id.nunique()} games with player records loaded • {logs.player.nunique()} players')
        mode=st.radio('Analysis', ['🎯 Individual player', '📋 Weekly prop rankings'],horizontal=True,key='prop_mode')
        rng=np.random.default_rng(int(seed))
        def prop_draws(vals,count):
            vals=np.asarray(vals,dtype=float)
            # Empirical bootstrap with light continuous noise for yards; nonnegative and integer for counts.
            picks=rng.choice(vals,size=int(count),replace=True)
            sigma=max(float(np.std(vals)),1.0)
            draws=np.maximum(0,picks+rng.normal(0,0.2*sigma,size=int(count)))
            return draws
        if mode.startswith('🎯'):
            player=st.selectbox('Player',sorted(logs.player.unique()),key='prop_player')
            choices=sorted(logs.loc[logs.player==player,'prop'].unique())
            prop=st.selectbox('Statistic',choices,key='prop_type')
            vals=logs[(logs.player==player)&(logs.prop==prop)].drop_duplicates('game_id').sort_values('game_id').value.to_numpy()
            line=st.number_input('Prop line',min_value=0.0,value=float(round(np.mean(vals)*2)/2),step=.5,key='prop_line')
            if len(vals)<3:st.warning('Fewer than 3 games: estimate is extremely unstable.')
            draws=prop_draws(vals,n)
            p_over=float(np.mean(draws>line));p_under=float(np.mean(draws<line))
            a,b,c=st.columns(3)
            a.metric('Over probability',f'{p_over:.1%}');b.metric('Under probability',f'{p_under:.1%}');c.metric('Mean projection',f'{draws.mean():.1f}')
            st.write(f'**Games sampled:** {len(vals)} • **Median:** {np.median(draws):.1f} • **80% simulated interval:** {np.quantile(draws,.1):.1f}–{np.quantile(draws,.9):.1f}')
            st.bar_chart(pd.Series(np.histogram(draws,bins=25)[0],index=np.round(np.histogram_bin_edges(draws,bins=25)[:-1],1)))
            st.caption('Uses an empirical bootstrap of available recent game logs, with small added noise. Does not account for injuries, opponent, projected snaps or sportsbook odds.')
        else:
            st.write('Rank players by simulated chance of exceeding a **user-adjustable baseline line** (the median of available game logs). These are not sportsbook lines or verified betting edges.')
            selected_prop=st.selectbox('Prop category',list(PROP_KEYS),key='rank_prop')
            minimum=st.slider('Minimum games recorded',3,10,4,key='min_prop_games')
            subset=logs[logs.prop==selected_prop]
            ranked=[]
            for player,group in subset.groupby('player'):
                vals=group.drop_duplicates('game_id').value.to_numpy()
                if len(vals)<minimum:continue
                baseline=round(float(np.median(vals))*2)/2
                sim=prop_draws(vals,min(n,10000))
                ranked.append({'Player':player,'Team':group.team.iloc[-1],'Games':len(vals),'Reference line (not sportsbook)':baseline,'Projection':round(float(sim.mean()),1),'Over %':round(float(np.mean(sim>baseline)*100),1),'Under %':round(float(np.mean(sim<baseline)*100),1)})
            if ranked:
                ranked_df=pd.DataFrame(ranked).sort_values('Over %',ascending=False)
                st.dataframe(ranked_df,use_container_width=True,hide_index=True)
                st.download_button('Download player prop rankings CSV',ranked_df.to_csv(index=False),'player_prop_rankings.csv','text/csv')
            else:st.warning('No players have enough games for this category. Try reducing the minimum games.')
    st.caption('Coverage is limited to the selected completed weeks. Rankings are NOT a scan of current sportsbook markets. The model requires validation and matchup adjustments before interpreting differences as edges.')

# ----------------- Animated, synthetic play-by-play game -----------------
st.divider()
st.header('🏟️ Live Game Theater')
st.caption('Watch one *fictional simulated game* unfold play-by-play. This is NOT a real NFL live feed or a reconstruction of actual plays.')

import html as _html

@st.cache_data(ttl=21600,show_spinner=False)
def _team_roster(team):
    """Resolve the ESPN team ID and parse both grouped and flat roster formats."""
    empty={'QB':[],'RB':[],'WR':[],'TE':[],'DEF':[]}
    team_id=team_id_map().get(team)
    if not team_id:
        return {**empty,'_error':f'Could not resolve ESPN team ID for {team}'}
    try:
        url=f'https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/roster'
        response=requests.get(url,params={'season':SEASON},timeout=12)
        response.raise_for_status()
        data=response.json()
        athletes=[]
        def collect(node):
            if isinstance(node,list):
                for item in node:collect(item)
            elif isinstance(node,dict):
                if node.get('displayName') or node.get('fullName'):
                    if node.get('position') or node.get('positions'):
                        athletes.append(node)
                for key in ('athletes','items','groups'):
                    if key in node:collect(node[key])
        collect(data.get('athletes',[]))
        roster={k:[] for k in empty}
        for athlete in athletes:
            pos=athlete.get('position') or {}
            if isinstance(pos,list):pos=pos[0] if pos else {}
            abbrev=pos.get('abbreviation','') if isinstance(pos,dict) else str(pos)
            name=athlete.get('displayName') or athlete.get('fullName')
            if not name:continue
            group=abbrev if abbrev in roster else ('DEF' if abbrev in ('DE','DT','LB','CB','S','FS','SS','OLB','ILB','NT') else None)
            if group and name not in roster[group]:roster[group].append(name)
        if not any(roster.values()):
            return {**empty,'_error':f'ESPN returned no recognized players for {team} (ID {team_id})'}
        return roster
    except (requests.RequestException,ValueError,TypeError,KeyError) as exc:
        return {**empty,'_error':f'{team}: {type(exc).__name__}: {exc}'}

def _pick_player(rng,roster,position):
    names=roster.get(position,[])
    if not names:return f'Roster unavailable ({position})'
    if position=='QB':return roster.get('_starting_qb') or names[0]
    weights=np.array([0.50**i for i in range(len(names))],dtype=float)
    overrides=roster.get('_usage_weights',{}).get(position,{})
    if overrides:
        weights=np.array([max(0.0,float(overrides.get(name,0))) for name in names],dtype=float)
        if not weights.any():weights=np.ones(len(names),dtype=float)
    weights/=weights.sum()
    return str(rng.choice(names,p=weights))

def _player_row(s,team,name):
    key=(team,name)
    if key not in s['player_stats']:
        s['player_stats'][key]={'Team':team,'Player':name,'Pass Att':0,'Completions':0,'Pass Yds':0,'Pass TD':0,'INT':0,'Carries':0,'Rush Yds':0,'Rush TD':0,'Targets':0,'Receptions':0,'Rec Yds':0,'Rec TD':0}
    return s['player_stats'][key]

def _new_theater(away, home, seed_value):
    return {'away':away,'home':home,'seed':int(seed_value),'clock':900,'quarter':1,
            'poss':home,'yard':25,'down':1,'to_go':10,'score':{away:0,home:0},
            'plays':[],'stats':{t:{'pass_yards':0,'rush_yards':0,'completions':0,'attempts':0,'rushes':0,'td':0,'turnovers':0} for t in [away,home]},
            'finished':False,'play_index':0,'player_stats':{},'rosters':{away:_team_roster(away),home:_team_roster(home)}}

def _advance_game(s):
    if s['finished']:return
    rng=np.random.default_rng(s['seed']+s['play_index']*7919)
    offense=s['poss']; defense=s['home'] if offense==s['away'] else s['away']
    r=rating.get(offense,(avg,avg,1)); d=rating.get(defense,(avg,avg,1))
    strength=np.clip((r[0]-d[1])/35,-.4,.4)
    q=s['quarter']; clock=s['clock']; start_y=s['yard']
    elapsed=int(rng.integers(18,44)); kind='pass' if rng.random()<.57 else 'rush'
    change=False; text=''; scored=False; scorer=None; score_kind=None
    roster=s['rosters'][offense]; qb=_pick_player(rng,roster,'QB')
    if rng.random()<.018:
        s['stats'][offense]['turnovers']+=1;change=True
        text=f'{offense}: turnover! Defense takes possession.'
    elif kind=='pass':
        s['stats'][offense]['attempts']+=1
        passer=_player_row(s,offense,qb);passer['Pass Att']+=1
        receiver=_pick_player(rng,roster,'WR' if rng.random()<.75 else ('TE' if rng.random()<.75 else 'RB'))
        target=_player_row(s,offense,receiver);target['Targets']+=1
        if rng.random()<np.clip(.63+strength*.2,.35,.83):
            yards=int(np.clip(rng.normal(9+strength*9,11),-7,65))
            s['stats'][offense]['completions']+=1;s['stats'][offense]['pass_yards']+=max(0,yards)
            passer['Completions']+=1;passer['Pass Yds']+=max(0,yards)
            target['Receptions']+=1;target['Rec Yds']+=max(0,yards)
            scorer=receiver;score_kind='pass'
            text=f'{qb} completes a pass to {receiver} for {yards} yards.'
        else:
            yards=0;text=f'{qb} throws incomplete targeting {receiver}.'
    else:
        yards=int(np.clip(rng.normal(4.5+strength*4,5),-5,35))
        runner=qb if rng.random()<.08 else _pick_player(rng,roster,'RB')
        row=_player_row(s,offense,runner);row['Carries']+=1;row['Rush Yds']+=max(0,yards)
        scorer=runner;score_kind='rush'
        s['stats'][offense]['rushes']+=1;s['stats'][offense]['rush_yards']+=max(0,yards)
        text=f'{runner} rushes for {yards} yards.'
    if not change:
        s['yard']=max(1,min(100,s['yard']+yards))
        if s['yard']>=100:
            s['score'][offense]+=7;s['stats'][offense]['td']+=1
            if scorer:
                row=_player_row(s,offense,scorer)
                if score_kind=='pass':row['Rec TD']+=1;_player_row(s,offense,qb)['Pass TD']+=1
                elif score_kind=='rush':row['Rush TD']+=1
            text+=' TOUCHDOWN! (+7, assuming successful kick)';scored=True;change=True
        elif yards>=s['to_go']:
            s['down']=1;s['to_go']=min(10,100-s['yard'])
        else:
            s['down']+=1;s['to_go']=max(1,s['to_go']-yards)
            if s['down']>4:
                if s['yard']>=63 and rng.random()<.74:
                    s['score'][offense]+=3;text+=' Field goal is GOOD!';scored=True
                else:text+=' Turnover on downs / missed kick.'
                change=True
    if change:
        s['poss']=defense;s['yard']=25 if scored else max(10,min(90,100-s['yard']))
        s['down']=1;s['to_go']=10
    s['clock']-=elapsed
    if s['clock']<=0:
        s['quarter']+=1;s['clock']=900
        text+=f' End of Q{q}.'
        if s['quarter']>4:
            # Sudden-death simplified tie-break; game must have winner.
            if s['score'][s['away']]==s['score'][s['home']]:
                winner=s['home'] if rng.random()<.5 else s['away']
                s['score'][winner]+=3;text+=f' Simulated overtime: {winner} wins on a field goal.'
            s['finished']=True
    s['play_index']+=1
    s['plays'].append(f"Q{min(q,4)} {clock//60:02d}:{clock%60:02d} · {text}")
    if s['play_index']>=200:s['finished']=True

def _field(s):
    # field is oriented from offense's own goal line toward opponent's goal line
    x=20+min(100,max(0,s['yard']))*5.6
    marks=''.join(f'<line x1="{20+i*56}" y1="15" x2="{20+i*56}" y2="170" stroke="#ffffff55" stroke-width="1" />' for i in range(11))
    nums=''.join(f'<text x="{20+i*56}" y="94" text-anchor="middle" fill="white" opacity=".65" font-size="13">{min(i,10-i)*10}</text>' for i in range(1,10))
    return f'''<svg viewBox="0 0 600 185" role="img" aria-label="Simulated field position: {s['yard']} yards from own goal" style="width:100%;background:#17683d;border-radius:14px"><rect x="20" y="15" width="560" height="155" fill="#17683d" stroke="white" stroke-width="2"/>{marks}{nums}<line x1="{x}" y1="15" x2="{x}" y2="170" stroke="#f9d448" stroke-width="4"/><circle cx="{x}" cy="48" r="11" fill="#a8612c" stroke="white" stroke-width="2"/><text x="300" y="155" fill="white" font-size="14" text-anchor="middle">{_html.escape(s['poss'])} possession • {s['down']} &amp; {s['to_go']}</text></svg>'''

with st.expander('🎮 Watch an animated simulated game', expanded=True):
    tc1,tc2=st.columns(2)
    theater_away=tc1.selectbox('Away team',teams,index=0,key='theater_away')
    theater_home=tc2.selectbox('Home team',[t for t in teams if t!=theater_away],index=0,key='theater_home')
    # Explicit lineup controls: ESPN roster order is not a verified depth chart.
    roster_a=_team_roster(theater_away)
    roster_h=_team_roster(theater_home)
    lineup_choices={}
    with st.expander('🧢 Set starters and player usage (recommended)',expanded=True):
        st.caption('ESPN roster order is not a depth chart. Choose the expected starting QB and optionally adjust RB/WR/TE usage. Check injury news before kickoff. Selections are applied on Kickoff / Reset.')
        for team_code, roster_data in ((theater_away,roster_a),(theater_home,roster_h)):
            st.markdown(f'**{team_code} lineup**')
            qbs=roster_data.get('QB',[])
            selected=st.selectbox(f'{team_code} starting QB',qbs if qbs else ['Roster unavailable'],key=f'qb_override_{team_code}')
            lineup_choices[team_code]={'_starting_qb':selected,'_usage_weights':{}}
            for position in ('RB','WR','TE'):
                names=roster_data.get(position,[])
                if not names:continue
                with st.expander(f'{team_code} {position} usage',expanded=False):
                    st.caption('Relative usage weights, not projected touches or targets. Set injured/inactive players to 0.')
                    vals={}
                    for i,name in enumerate(names):
                        vals[name]=st.number_input(name,min_value=0,max_value=100,value=max(1,int(100*(0.5**i))),step=5,key=f'usage_{team_code}_{position}_{name}')
                    lineup_choices[team_code]['_usage_weights'][position]=vals
    pace=st.select_slider('Playback',options=['Manual','1 play / 3 sec','1 play / sec','Fast (4 plays / sec)'],value='Manual',key='theater_speed')
    if st.button('🏈 Kickoff / Reset',key='theater_reset'):
        _team_roster.clear()
        st.session_state.theater_playing=False
        st.session_state.theater_state=_new_theater(theater_away,theater_home,int(seed))
        for code,choice in lineup_choices.items():st.session_state.theater_state['rosters'][code].update(choice)
    if 'theater_state' not in st.session_state:
        st.session_state.theater_state=_new_theater(theater_away,theater_home,int(seed))
    state=st.session_state.theater_state
    if state['away']!=theater_away or state['home']!=theater_home:
        state=_new_theater(theater_away,theater_home,int(seed));st.session_state.theater_state=state
        for code,choice in lineup_choices.items():state['rosters'][code].update(choice)
    if 'theater_playing' not in st.session_state:st.session_state.theater_playing=False
    if pace=='Manual':st.session_state.theater_playing=False
    b1,b2,b3=st.columns(3)
    if b1.button('▶️ Play / Pause',key='theater_toggle'):
        st.session_state.theater_playing=not st.session_state.theater_playing
    if b2.button('⏭️ Next play',key='theater_next'):_advance_game(state)
    if b3.button('⏩ Finish game',key='theater_finish'):
        while not state['finished']:_advance_game(state)
        st.session_state.theater_playing=False
    period={'Manual':None,'1 play / 3 sec':3,'1 play / sec':1,'Fast (4 plays / sec)':0.25}[pace]
    @st.fragment(run_every=period if period is not None else None)
    def render_theater():
        s=st.session_state.theater_state
        if st.session_state.theater_playing and period is not None and not s['finished']:
            _advance_game(s)
        if s['finished']:st.session_state.theater_playing=False
        st.markdown(f"### {s['away']} **{s['score'][s['away']]}** — **{s['score'][s['home']]}** {s['home']}")
        st.caption('FINAL' if s['finished'] else f"Q{s['quarter']} • {s['clock']//60:02d}:{s['clock']%60:02d} remaining • Play #{s['play_index']}")
        st.markdown(_field(s),unsafe_allow_html=True)
        st.markdown('**Latest play**')
        st.info(s['plays'][-1] if s['plays'] else 'Kickoff ready. Press Next play, or select a playback speed and Play.')
        st.markdown('**Live synthetic team statistics**')
        st.dataframe(pd.DataFrame.from_dict(s['stats'],orient='index').rename_axis('Team').reset_index(),hide_index=True,use_container_width=True)
        st.markdown('**Live simulated player box score**')
        if s['player_stats']:
            st.dataframe(pd.DataFrame(s['player_stats'].values()),hide_index=True,use_container_width=True)
        else:st.caption('Player statistics will appear after the first pass or run.')
        st.markdown('**Play-by-play feed**')
        st.text('\n'.join(reversed(s['plays'][-12:])) if s['plays'] else 'No plays yet')
        for team_code, roster_data in s['rosters'].items():
            if roster_data.get('_error'):
                st.error(f'Roster unavailable for {team_code}: {roster_data["_error"]}. Player names cannot be verified. Check the data feed and restart the simulation.')
        st.caption('Fictional simulation using best-effort roster names from ESPN; roster order is NOT a verified depth chart; set the starters above. Unlisted positions are labeled. Player usage and outcomes are synthetic, not player-specific forecasts. Field animation is a marker, not moving athletes. Simplified clock, turnovers, and scoring rules.')
    render_theater()

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import pandas as pd

    return alt, mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Hyperparameter tuning for Multimodal Transformers for Humor Recognition
    """)
    return


@app.cell
def _(pd):
    _models_names = {
        'bert-base-portuguese-cased': 'BERTimbau Base',
        'bert-large-portuguese-cased': 'BERTimbau Large',
        'albertina-900m-portuguese-ptbr-encoder': 'Albertina PT-BR',
        'albertina-900m-portuguese-ptpt-encoder': 'Albertina PT-PT'
    }
    _models_order = ['BERTimbau Base', 'BERTimbau Large', 'Albertina PT-BR', 'Albertina PT-PT']
    _methods_order = ['concatenation', 'pooling', 'shared']

    sweeps_df = pd.read_parquet('results/sweeps/sweeps.parquet')
    sweeps_df['model'] = sweeps_df['model'].replace(_models_names)
    sweeps_df['model'] = pd.Categorical(sweeps_df['model'], categories=_models_order, ordered=True)
    sweeps_df['method'] = pd.Categorical(sweeps_df['method'], categories=_methods_order, ordered=True)
    runs_df = pd.read_parquet('results/sweeps/runs.parquet')
    history_df = pd.read_parquet('results/sweeps/history.parquet')
    best_df = (
        history_df.dropna(subset=['eval/f1_macro'])
        .groupby('run name', as_index=False)['eval/f1_macro'].max()
    )
    summary_df = best_df.merge(runs_df, on='run name').merge(sweeps_df, on='sweep id')
    return history_df, summary_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What are the best models?
    """)
    return


@app.cell
def _(mo, summary_df):
    mo.ui.table(
        (
            summary_df.groupby(['model', 'method'], as_index=False, observed=True)['eval/f1_macro'].max()
            [['method', 'model', 'eval/f1_macro']]
            .sort_values(by=['method', 'model'])
        ),
        selection=None,
        pagination=False,
        format_mapping={
            'eval/f1_macro': '{:.2f}'
        }
    )
    return


@app.cell
def _(alt, mo, summary_df):
    _heatmap_data = summary_df.groupby(['model', 'method'], as_index=False, observed=True)['eval/f1_macro'].max()

    _base = (
        alt.Chart(_heatmap_data)
        .encode(
            x=alt.X('method:N', title='Fusion method'),
            y=alt.Y('model:N', title='Base model')
        )
        .properties(
            width=500,
            height=400
        )
    )
    _heatmap = _base.mark_rect().encode(
        color=alt.Color('eval/f1_macro:Q', scale=alt.Scale(scheme='redyellowgreen'), title='Max F1 Macro')
    )
    _text = _base.mark_text(baseline='middle').encode(
        alt.Text('eval/f1_macro:Q', format='.2f')
    )

    _g = _heatmap + _text
    mo.ui.altair_chart(_g)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analysis for each hyperparameter
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Freeze base model
    """)
    return


@app.cell
def _(alt, mo, summary_df):
    _dumbbell_data = summary_df.groupby(['model', 'method', 'freeze_base'], as_index=False, observed=True)['eval/f1_macro'].max()
    _dumbbell_data['architecture'] = _dumbbell_data['model'].astype(str) + ' | ' + _dumbbell_data['method'].astype(str)
    _dumbbell_data = _dumbbell_data.sort_values(['method', 'model'])

    _pivot_data = _dumbbell_data.pivot(index=['model', 'method', 'architecture'], columns='freeze_base', values='eval/f1_macro')
    _pivot_data = _pivot_data.rename(columns={True: 'frozen_f1', False: 'unfrozen_f1'})
    _pivot_data = _pivot_data.reset_index()
    _pivot_data['delta'] = _pivot_data['unfrozen_f1'] - _pivot_data['frozen_f1']
    _pivot_data['midpoint'] = (_pivot_data['unfrozen_f1'] + _pivot_data['frozen_f1']) / 2
    _pivot_data = _pivot_data.sort_values(['method', 'model'])


    _lines = (
        alt.Chart(_pivot_data)
        .mark_rule(color='gray', strokeWidth=2)
        .encode(
            alt.Y('architecture:N', sort=None, title='Model | Method'),
            alt.X('unfrozen_f1:Q', title='Max F1'),
            alt.X2('frozen_f1:Q')
        )
    )
    _points = (
        alt.Chart(_dumbbell_data)
        .mark_circle(size=100)
        .encode(
            alt.Y('architecture:N', sort=None),
            alt.X('eval/f1_macro:Q', scale=alt.Scale(zero=False)),
            alt.Color('freeze_base', title='Is base model frozen?')
        )
    )
    _text = (
        alt.Chart(_pivot_data)
        .transform_calculate(
            delta_label="datum.delta > 0 ? '+' + format(datum.delta, '.3f') : format(datum.delta, '.3f')"
        )
        .mark_text(
            baseline='bottom',
            dy=-2
        )
        .encode(
            alt.Y('architecture:N', sort=None),
            alt.X('midpoint:Q'),
            alt.Text('delta_label:N')
        )
    )

    _g = (_lines + _points + _text).properties(
        width=400,
        height=350
    )

    mo.ui.altair_chart(_g)
    return


@app.cell
def _(alt, mo, summary_df):
    _g = (
        alt.Chart(summary_df)
        .mark_circle(size=60, opacity=0.7)
        .encode(
            x=alt.X('learning_rate:Q', scale=alt.Scale(type='log'), title='Learning Rate (Log scale)'),
            y=alt.Y('eval/f1_macro:Q', title='Max F1 Macro'),
            color=alt.Color('freeze_base:N', title='Is base model frozen?')
        )
        .properties(
            width=600,
            height=300
        )
        .interactive()
    )
    mo.ui.altair_chart(_g)
    return


@app.cell
def _(alt, mo, summary_df):
    _g = (
        alt.Chart(summary_df)
        .mark_boxplot(extent='min-max')
        .encode(
            x=alt.X('method:N', sort='-y', title='Fusion method'),
            y=alt.Y('eval/f1_macro:Q', scale=alt.Scale(zero=False), title='F1 Macro'),
            color=alt.Color('model:N', title='Base model'),
            xOffset=alt.XOffset('model:N')
        )
        .properties(
            width = 500,
            height = 300
        )
    )

    mo.ui.altair_chart(_g)
    return


@app.cell
def _(alt, history_df, mo, summary_df):
    _top_runs = summary_df.nlargest(5, 'eval/f1_macro')['run name'].tolist()
    _top_history = history_df[history_df['run name'].isin(_top_runs)].dropna(subset=['eval/f1_macro'])
    _top_history = _top_history.merge(summary_df[['run name', 'model', 'method']], on='run name')
    _top_history['Run label'] = _top_history['model'].astype(str) + ' + ' + _top_history['method'].astype(str) + ' (' + _top_history['run name'] + ')'

    _g = (
        alt.Chart(_top_history)
        .mark_line(point=True)
        .encode(
            x=alt.X('_step:Q'),
            y=alt.Y('eval/f1_macro:Q', scale=alt.Scale(zero=False)),
            color=alt.Color('Run label:N')
        )
        .properties(
            width=800,
            height=400
        )
    )

    mo.ui.altair_chart(_g)
    return


if __name__ == "__main__":
    app.run()

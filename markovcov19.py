if __name__ == '__main__':
    import sys
    sys.path.append('../')
    import pandas as pd
    import matplotlib.pyplot as plt


    import pyncov as nc
    import pyncov.io
    import pyncov.plot
    import pyncov.datasets
    import numpy as np

    import pyncov.datasets
 	
    print('pyncov-19', nc.__version__)
    print('numpy', np.__version__)
    print('pandas', pd.__version__)

    try:
        plt.style.use('seaborn-whitegrid')
    except:
        print('Using default style')

    figsize = (16, 6)

    # Configuration of the markov chain
    n = 100000
    bins = 100
    m = nc.build_markovchain(nc.MARKOV_DEFAULT_PARAMS)
    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Transmission times between the states of the model", fontsize=24)
    pd.Series(m.timeSimulator[nc.S.I1, nc.S.I2](n)).plot.hist(bins=bins, ax=ax[0,0], title='I1 to I2');
    pd.Series(m.timeSimulator[nc.S.I1, nc.S.I3](n)).plot.hist(bins=bins, ax=ax[0,1], title='I1 to I3');
    pd.Series(m.timeSimulator[nc.S.I3, nc.S.M0](n)).plot.hist(bins=bins, ax=ax[0,2], title='I3 to M');
    pd.Series(m.timeSimulator[nc.S.I3, nc.S.R1](n)).plot.hist(bins=bins, ax=ax[1,0], title='I3 to R1');
    pd.Series(m.timeSimulator[nc.S.R1, nc.S.R2](n)).plot.hist(bins=bins, ax=ax[1,1], title='R1 to R2');
    pd.Series(m.timeSimulator[nc.S.I2, nc.S.R1](n)).plot.hist(bins=bins, ax=ax[1,2], title='I2 to R1');

    infection_rates = nc.infection_rates([0.8, 0.85, 12, 0.01], num_days=60)
    pd.Series(infection_rates).plot(title="Dynamic individual infection rate $R_i(t)$");


    num_chains = 100
    initial_infections = 1
    susceptible_population = 999999
    population = initial_infections + susceptible_population
    num_days = 60

    simulations = nc.sample_chains(susceptible_population, initial_infections, m, infection_rates, num_chains=num_chains, n_workers=4, show_progress=True)

    # The dimensions are num_chains x num_days x states
    simulations.shape

    # Get the new infections of the first simulation
    simulations[0,:,nc.S.I1].astype(int)
    fig, ax = plt.subplots(figsize=figsize)
    pd.Series(simulations[0,:,nc.S.I1], name='Infections').plot.bar(ax=ax, legend=True);
    pd.Series(simulations[0,:,nc.S.M0], name='Fatalities').plot.bar(ax=ax, color='r', legend=True);


    fig, ax = plt.subplots(1, 3, figsize=(16,4))
    nc.plot.plot_state(simulations, nc.S.I1, ax=ax[0], title="New infections over time")
    nc.plot.plot_state(simulations, nc.S.M0, diff=True, ax=ax[1], title="Daily deaths")
    nc.plot.plot_state(simulations, nc.S.M0, ax=ax[2], title="Total deaths")

    df_global = nc.datasets.load_csse_global_fatalities()
    # Get only country level
    df_global = df_global[df_global.index.get_level_values('n2').isna()]
    df_global.head()

    df_global.loc[['GBR','IND','USA']].transpose().plot(logy=True)

    df_india = nc.datasets.load_csse_global_fatalities().query("n1 == 'India'")
    susceptible = df_india.index.get_level_values('population').values[0]
    df_india.squeeze().plot(title='Total fatalities')



    df = df_india.squeeze()
    # Remove some zeros to align the data
    df = df.reindex(pd.date_range(df.index[0] + 23 * df.index.freq, df.index[-1]), fill_value=0)

    infection_rates = nc.infection_rates(nc.io.get_trained_params('ESP'), num_days=len(df.index))
    pd.Series(infection_rates, index=df.index).plot(title="Dynamic individual infection rate $R_i(t)$ in Delhi");

    # Simulate using the population from Delhi (~30.300.000)
    # Use 4 processed to parallelize the generation of the chains
    simulations = nc.sample_chains(susceptible, initial_infections, m, infection_rates, num_chains=100, n_workers=4, show_progress=True)


    fig, ax = plt.subplots(1, 3, figsize=(16,4))
    nc.plot.plot_state(simulations, nc.S.I1, ax=ax[0], index=df.index, title="New infections over time")
    nc.plot.plot_state(simulations, nc.S.M0, diff=True, ax=ax[1], index=df.index, title="Fatalities")
    nc.plot.plot_state(simulations, nc.S.M0, ax=ax[2], index=df.index, title="Fatalities")
    df.diff().plot(ax=ax[1])
    df.plot(ax=ax[2])

    plt.show()

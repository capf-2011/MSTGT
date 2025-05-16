from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/gw/research/SOT/got10k_lmdb'
    settings.got10k_path = '/home/gw/research/SOT/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/gw/research/SOT/itb'
    settings.lasot_extension_subset_path = '/home/gw/Data/SOT/LaSOT_Ext'
    settings.lasot_lmdb_path = '/home/gw/research/SOT/lasot_lmdb'
    settings.lasot_path = '/home/gw/research/SOT/lasot'
    settings.network_path = '/home/gw/Data/exp_results/MSTGT/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/gw/research/SOT/nfs'
    settings.otb_path = '/home/gw/Data/SOT/OTB100'
    settings.prj_dir = '/home/gw/research/Tracking/pycharm_project/MSTGT'
    settings.result_plot_path = '/home/gw/Data/exp_results/MSTGT/output/test/result_plots'
    settings.results_path = '/home/gw/Data/exp_results/MSTGT/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/gw/Data/exp_results/MSTGT/output'
    settings.segmentation_path = '/home/gw/Data/exp_results/MSTGT/output/test/segmentation_results'
    settings.tc128_path = '/home/gw/research/SOT/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/gw/Data/SOT/TNL2K_CVPR2021'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/gw/research/SOT/trackingnet'
    settings.uav_path = '/home/gw/Data/SOT/UAV123'
    settings.vot18_path = '/home/gw/research/SOT/vot2018'
    settings.vot22_path = '/home/gw/research/SOT/vot2022'
    settings.vot_path = '/home/gw/research/SOT/VOT2019'
    settings.youtubevos_dir = ''

    return settings


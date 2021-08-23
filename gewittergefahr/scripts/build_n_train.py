import argparse

"""
in_files=None,parallel=True,pb=True,verbose=True,
                 ChaseFiles=True,scale=True,fillval=0.0,fillnan=True,
                 label_idx=2,training_dts = [np.datetime64('2013-01-01'),np.datetime64('2018-01-01')],
                 testing_dts = [np.datetime64('2018-01-01'),np.datetime64('2020-01-01')],
                 image_size = 32, batch_size=32,scale_of_noise=0.1,mean_of_noise=0,
                 rotation_ratio_radians = 0.08333333333333333,random_seed = None,
                 n_heights=20,n_channels=7 """"
            
CHASE_FILES_ARG_NAME = 'learning_example_file'
SCALE_ARG_NAME = 'storm_image_dir'
LEVEL_TO_PLOT_ARG_NAME = 'level'
LINKAGE_DIR_ARG_NAME = 'linkage_dir'
SEGMOTION_DIR_ARG_NAME = 'seg_dir'
GRIDRAD_DIR_ARG_NAME = 'rad_dir'
NEXRAD_LOC_ARG_NAME = 'nexrad_loc_csv'
SAVE_DIR_ARG_NAME = 'save_dir'
SAVEFIG_BOOL_ARG_NAME = 'savefig'
ALTER_FILES_BOOL_ARG_NAME = 'alterfiles'

LEARNING_EXAMPLE_FILE_HELP_STRING = (
    'file you wish to verify')

STORM_IMAGE_DIR_HELP_STRING = (
    'directory path where storm images are.')

LEVEL_TO_PLOT_HELP_STRING = (
    'Which height of radar data to plot')

LINKAGE_DIR_HELP_STRING = (
    'directory path where linked files are')

SEGMOTION_DIR_HELP_STRING = (
    'directory path where segmotion tracking files are')

GRIDRAD_DIR_HELP_STRING = (
    'directory path where gridded gridrad files are')

NEXRAD_LOC_HELP_STRING = (
    'Location of nexrad locations csv file')

SAVE_DIR_HELP_STRING = (
    'Path of where to save the .png images')

SAVEFIG_BOOL_HELP_STRING = (
    'Turn on or off the saving of the .pngs')

ALTER_FILES_BOOL_HELP_STRING = (
    'Turn on off the adding of extra metadata')

INPUT_ARG_PARSER = argparse.ArgumentParser()

INPUT_ARG_PARSER.add_argument(
    '--' + LEARNING_EXAMPLE_FILE_ARG_NAME, type=str, required=True,
    default='', help=LEARNING_EXAMPLE_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + STORM_IMAGE_DIR_ARG_NAME, type=str, required=True,
    default='',
    help=STORM_IMAGE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LEVEL_TO_PLOT_ARG_NAME, type=str,required=False,
    default='04000_metres_agl', help=LEVEL_TO_PLOT_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LINKAGE_DIR_ARG_NAME, type=str, required=True,
    default='',
    help=LINKAGE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SEGMOTION_DIR_ARG_NAME, type=str, required=True,
    default='',
    help=SEGMOTION_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + GRIDRAD_DIR_ARG_NAME, type=str,required=True,
    default='',
    help=GRIDRAD_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NEXRAD_LOC_ARG_NAME, type=str, required=True,
    help=NEXRAD_LOC_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SAVE_DIR_ARG_NAME, type=str, required=True,
    default='', help=SAVE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SAVEFIG_BOOL_ARG_NAME, type=bool, required=False,default=True,
    help=SAVEFIG_BOOL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + ALTER_FILES_BOOL_ARG_NAME, type=bool, required=False,
    default=False, help=ALTER_FILES_BOOL_HELP_STRING)

LEARNING_EXAMPLE_FILE_ARG_NAME = 'learning_example_file'
STORM_IMAGE_DIR_ARG_NAME = 'storm_image_dir'
LEVEL_TO_PLOT_ARG_NAME = 'level'
LINKAGE_DIR_ARG_NAME = 'linkage_dir'
SEGMOTION_DIR_ARG_NAME = 'seg_dir'
GRIDRAD_DIR_ARG_NAME = 'rad_dir'
NEXRAD_LOC_ARG_NAME = 'nexrad_loc_csv'
SAVE_DIR_ARG_NAME = 'save_dir'
SAVEFIG_BOOL_ARG_NAME = 'savefig'
ALTER_FILES_BOOL_ARG_NAME = 'alterfiles'

if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()
    ds_images = validate_examples(input_example_filename=getattr(INPUT_ARG_OBJECT, LEARNING_EXAMPLE_FILE_ARG_NAME),
                                  storm_image_dir=getattr(INPUT_ARG_OBJECT, STORM_IMAGE_DIR_ARG_NAME),
                                  level=getattr(INPUT_ARG_OBJECT, LEVEL_TO_PLOT_ARG_NAME),
                                  linkage_dir=getattr(INPUT_ARG_OBJECT, LINKAGE_DIR_ARG_NAME),
                                  seg_dir=getattr(INPUT_ARG_OBJECT, SEGMOTION_DIR_ARG_NAME),
                                  rad_dir=getattr(INPUT_ARG_OBJECT, GRIDRAD_DIR_ARG_NAME),
                                  nexrad_loc_csv=getattr(INPUT_ARG_OBJECT, NEXRAD_LOC_ARG_NAME),
                                  save_dir=getattr(INPUT_ARG_OBJECT, SAVE_DIR_ARG_NAME),
                                  savefig=getattr(INPUT_ARG_OBJECT, SAVEFIG_BOOL_ARG_NAME),
                                  alterfiles=getattr(INPUT_ARG_OBJECT, ALTER_FILES_BOOL_ARG_NAME),)

from unittest import TestCase
from unittest.mock import patch
from cloe.user_interface.likelihood_ui import LikelihoodUI


class LikelihoodUI_test(TestCase):

    def setUp(self):
        self.file_name = '/dev/null'
        # does not have a key named backend
        self.config_no_backend_key = {
            'name': 'anything', 'type': 'first'
        }
        # the backend is invalid
        self.config_backend_invalid = {
            'backend': 'invalid', 'type': 'first'
        }
        # the backend is Cosmosis
        self.config_backend_cosmosis = {
            'backend': 'Cosmosis', 'type': 'first'
        }
        # has a key/value pair that is backend/Cobaya, but misses a key named
        # Cobaya
        self.config_no_cobaya_key = {
            'backend': 'Cobaya', 'NotCobaya': 'any',
        }
        # has a key/value pair that is backend/Cobaya, and a key named Cobaya
        self.config_good = {
            'backend': 'Cobaya',
            'Cobaya': {'params': 'file.yaml',
                       'likelihood': {'Euclid': {'NL_flag_phot_matter': 0,
                                                 'NL_flag_spectro': 0,
                                                 'observables_selection': {},
                                                 'observables_specification':
                                                     {},
                                                 'data': '/dev/null'}},
                       'theory': {'camb': {}}},
            'Cosmosis': {},
            'action': 'run'
        }
        self.config_cosmosis = {
            'backend': 'Cosmosis',
            'Cobaya': {},
            'Cosmosis': {'params': 'file.yaml',
                         'likelihood': {'Euclid': {'NL_flag_phot_matter': 0,
                                                   'NL_flag_spectro': 0,
                                                   'observables_selection': {},
                                                   'observables_specification':
                                                   {},
                                                   'data': '/dev/null'}},
                         'theory': {'camb': {}}},
            'action': 'run'
        }
        self.config_action_invalid = {
            'backend': 'Cobaya',
            'Cobaya': {'params': 'file.yaml',
                       'likelihood': {'Euclid': {'NL_flag_phot_matter': 0,
                                                 'NL_flag_spectro': 0,
                                                 'observables_selection': {},
                                                 'observables_specification':
                                                     {},
                                                 'data': '/dev/null'}}},
            'Cosmosis': {},
            'action': 'invalid'}
        # for testing _update_config(), since it contains a nested dictionary
        # and a plain key/value pair, this should be sufficient to achieve
        # full coverage
        self.config_nested = {
            'key1': {'key1_1': 'val1'},
            'key2': 'val2',
            'key3': 'val3'
        }
        self.config_nested_update = {
            'key1': {'key1_1': 'val1-Updated'},
            'key2': 'val2-Updated'
        }
        self.config_nested_updated = {
            'key1': {'key1_1': 'val1-Updated'},
            'key2': 'val2-Updated',
            'key3': 'val3'
        }

    def tearDown(self):
        self.file_name = None
        self.config_no_backend_key = None
        self.config_backend_invalid = None
        self.config_backend_cosmosis = None
        self.config_no_cobaya_key = None
        self.config_good = None
        self.config_nested = None
        self.config_nested_update = None
        self.config_nested_updated = None

    # test init when the key specifying the backend is not present in
    # the input configuration
    @patch('cloe.auxiliary.logger.log_info')
    @patch('cloe.auxiliary.yaml_handler.yaml_read_and_check_dict')
    def test_init_no_backend_key(self, yaml_read_mock, log_mock):
        yaml_read_mock.return_value = self.config_no_backend_key
        self.assertRaises(KeyError,
                          LikelihoodUI,
                          user_config_file=self.file_name)
        self.assertEqual(yaml_read_mock.call_count, 2)

    # test initialization when the requested backend is not valid
    @patch('cloe.auxiliary.logger.log_info')
    @patch('cloe.auxiliary.yaml_handler.yaml_read_and_check_dict')
    def test_init_backend_invalid(self, yaml_read_mock, log_mock):
        yaml_read_mock.return_value = self.config_backend_invalid
        self.assertRaises(ValueError,
                          LikelihoodUI,
                          user_config_file=self.file_name)
        self.assertEqual(yaml_read_mock.call_count, 2)

    # test init when the requested backend is cosmosis (supported but not
    # implemented)
    @patch('cloe.auxiliary.logger.log_info')
    @patch('cloe.auxiliary.yaml_handler.yaml_read_and_check_dict')
    def test_init_backend_cosmosis(self, yaml_read_mock, log_mock):
        yaml_read_mock.return_value = self.config_backend_cosmosis
        self.assertEqual(yaml_read_mock.call_count, 0)

    # test behavior when the requested backend is cobaya
    @patch('cloe.auxiliary.logger.log_info')
    @patch('cloe.auxiliary.yaml_handler.yaml_read')
    @patch('cloe.auxiliary.likelihood_yaml_handler'
           '.update_cobaya_params_from_model_yaml')
    @patch('cloe.auxiliary.params_converter'
           '.convert_params')
    @patch('cloe.auxiliary.likelihood_yaml_handler'
           '.update_cobaya_dict_with_halofit_version')
    @patch('cobaya.run')
    def test_run_backend_cobaya(
            self,
            cobaya_run_mock,
            halofit_update_mock,
            convert_params_mock,
            dict_update_mock,
            yaml_read_mock,
            log_mock
    ):
        yaml_read_mock.return_value = self.config_good
        ui = LikelihoodUI(user_config_file=self.file_name)
        ui.run()
        self.assertEqual(yaml_read_mock.call_count, 3)
        self.assertEqual(dict_update_mock.call_count, 1)
        self.assertEqual(convert_params_mock.call_count, 1)
        self.assertEqual(halofit_update_mock.call_count, 1)
        self.assertEqual(cobaya_run_mock.call_count, 1)

    # test behavior of __init__() when the LikelihoodUI object is
    # instantiated with no arguments
    @patch('cloe.auxiliary.logger.log_info')
    @patch('cloe.auxiliary.yaml_handler.yaml_read')
    @patch('cloe.auxiliary.likelihood_yaml_handler'
           '.update_cobaya_params_from_model_yaml')
    @patch('cloe.auxiliary.params_converter'
           '.convert_params')
    @patch('cloe.auxiliary.likelihood_yaml_handler'
           '.update_cobaya_dict_with_halofit_version')
    @patch('cobaya.run')
    def test_init_no_args(
            self,
            cobaya_run_mock,
            halofit_update_mock,
            convert_params_mock,
            dict_update_mock,
            yaml_read_mock,
            log_mock
    ):
        yaml_read_mock.return_value = self.config_good
        ui = LikelihoodUI()
        ui.run()
        self.assertEqual(yaml_read_mock.call_count, 2)
        self.assertEqual(dict_update_mock.call_count, 1)
        self.assertEqual(convert_params_mock.call_count, 1)
        self.assertEqual(halofit_update_mock.call_count, 1)
        self.assertEqual(cobaya_run_mock.call_count, 1)

    # test behavior of the run_cobaya() private method when the input
    # configuration does not contain a key named 'Cobaya'.
    # Verify that an exception is thrown in this case
    @patch('cloe.auxiliary.logger.log_info')
    @patch('cloe.auxiliary.yaml_handler.yaml_read_and_check_dict')
    @patch('cobaya.run')
    def test_run_cobaya_no_cobaya_key(
            self,
            cobaya_run_mock,
            yaml_read_mock,
            log_mock):
        yaml_read_mock.return_value = self.config_no_cobaya_key
        ui = LikelihoodUI(user_config_file=self.file_name)
        self.assertRaises(KeyError, ui._run_cobaya)
        self.assertEqual(cobaya_run_mock.call_count, 0)

    # test run_cobaya private method: verify that the proper external calls
    # are performed and with the proper arguments
    @patch('cloe.auxiliary.logger.log_info')
    @patch('cloe.auxiliary.yaml_handler.yaml_read')
    @patch('cloe.auxiliary.likelihood_yaml_handler'
           '.update_cobaya_params_from_model_yaml')
    @patch('cloe.auxiliary.params_converter'
           '.convert_params')
    @patch('cloe.auxiliary.likelihood_yaml_handler'
           '.update_cobaya_dict_with_halofit_version')
    @patch('cobaya.run')
    def test_run_cobaya_success(
            self,
            cobaya_run_mock,
            halofit_update_mock,
            convert_params_mock,
            dict_update_mock,
            yaml_read_mock,
            log_mock
    ):
        yaml_read_mock.return_value = self.config_good
        dict_update_mock.return_value = self.config_good['Cobaya']
        ui = LikelihoodUI(user_config_file=self.file_name)
        ui._run_cobaya()
        self.assertEqual(dict_update_mock.call_count, 1)
        self.assertEqual(convert_params_mock.call_count, 1)
        self.assertEqual(halofit_update_mock.call_count, 1)
        self.assertEqual(cobaya_run_mock.call_count, 1)
        cobaya_run_mock.assert_called_with(self.config_good['Cobaya'])

    # update a dictionary containing a nested dictionary and a plain key/value
    # pair and verify that the updated dictionary matches the expectations.
    def test_run_update_config_not_recursive(self):
        updated = (
            LikelihoodUI._update_config(
                orig_config=self.config_nested,
                update_config=self.config_nested_update
            )
        )
        self.assertDictEqual(self.config_nested_updated, updated)

    def test_get_and_check_no_backend_key(self):
        self.assertRaises(KeyError,
                          LikelihoodUI._get_and_check_backend,
                          self.config_no_backend_key)

    def test_get_and_check_backend_cobaya(self):
        backend = LikelihoodUI._get_and_check_backend(self.config_good)
        self.assertEqual(backend, 'Cobaya')

    def test_get_and_check_backend_cosmosis(self):
        backend = LikelihoodUI._get_and_check_backend(self.config_cosmosis)
        self.assertEqual(backend, 'Cosmosis')

    def test_get_and_check_backend_invalid(self):
        self.assertRaises(ValueError,
                          LikelihoodUI._get_and_check_backend,
                          self.config_backend_invalid)

    @patch('cloe.auxiliary.logger.log_info')
    @patch('cloe.auxiliary.yaml_handler.yaml_read')
    @patch('cloe.auxiliary.likelihood_yaml_handler'
           '.update_cobaya_params_from_model_yaml')
    def test_get_and_check_valid_action(
            self,
            dict_update_mock,
            yaml_read_mock,
            log_mock
    ):
        yaml_read_mock.return_value = self.config_good
        dict_update_mock.return_value = self.config_good['Cobaya']
        ui = LikelihoodUI(user_config_file=self.file_name)
        action = ui.get_and_check_action()
        self.assertEqual(action, 'run')

    @patch('cloe.auxiliary.logger.log_info')
    @patch('cloe.auxiliary.yaml_handler.yaml_read')
    @patch('cloe.auxiliary.likelihood_yaml_handler'
           '.update_cobaya_params_from_model_yaml')
    def test_get_and_check_invalid_action(
            self,
            dict_update_mock,
            yaml_read_mock,
            log_mock
    ):
        yaml_read_mock.return_value = self.config_action_invalid
        dict_update_mock.return_value = self.config_action_invalid['Cobaya']
        ui = LikelihoodUI(user_config_file=self.file_name)
        self.assertRaises(ValueError, ui.get_and_check_action)

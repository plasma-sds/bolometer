import numpy as np
import pytest

from axuv.adf15 import read_adf15

# pec40 layout (W): lower-case block headers without INDM, an energy level
# table before the transition table, two lines from the same upper level
PEC40 = """\
    2    /W : 7 photon emissivity coefficients/
--------------------------------------------------------------------------------
    1    1
--------------------------------------------------------------------------------
  120.310    2    3 /pl=01:ss= 7:pb= 1:sz= 7/type=excit/ispb = t/isel =    1
 1.00E+10 1.00E+12
 1.00E+00 1.00E+01 1.00E+02
 1.00E-12 2.00E-12 3.00E-12
 4.00E-12 5.00E-12 6.00E-12
  133.398    2    3 /pl=01:ss= 7:pb= 1:sz= 7/type=excit/ispb = t/isel =    2
 1.00E+10 1.00E+12
 1.00E+00 1.00E+01 1.00E+02
 1.00D-13 2.00D-13 3.00D-13
 4.00D-13 5.00D-13 6.00D-13
C  energy levels
C   lv      configuration    (2S+1)L(w-1/2)      energy (cm^-1)
C  ---   ------------------- --------------      --------------
c    1   60963A52B56C           (2)3(  3.5)             0.0
c    2   60963A52B56C           (2)3(  2.5)         16726.9
C   isel  wvlen(A)            transition                type  ispb nspb
C  ----- ---------- ----------------------------------- ----- ---- ---- --
C      1   120.310  487(2)3(  3.5)-  1(2)3(  3.5)       excit   1    1   7   345 45  1
C      2   133.398  487(2)3(  3.5)-  2(2)3(  2.5)       excit   1    1   7   246 42  2
"""

# pec93 layout (B, llr): upper-case headers with INDM, an excitation block
# driven from metastable 2, and a recombination block from the next ion
PEC93 = """\
    3    /B + 3 PHOTON EMISSIVITY COEFFICIENTS/
    61.1 A   2   3 /FILMEM = WJD91#B /TYPE = EXCIT    /INDM = 1/ISEL =    1
 1.00E+10 1.00E+12
 1.00E+00 1.00E+01 1.00E+02
 1.00E-12 2.00E-12 3.00E-12
 4.00E-12 5.00E-12 6.00E-12
    61.1 A   2   3 /FILMEM = WJD91#B /TYPE = EXCIT    /INDM = 2/ISEL =    2
 1.00E+10 1.00E+12
 1.00E+00 1.00E+01 1.00E+02
 9.00E-12 9.00E-12 9.00E-12
 9.00E-12 9.00E-12 9.00E-12
    61.1 A   2   3 /FILMEM = WJD91#B /TYPE = RECOM    /INDM =+1/ISEL =    3
 1.00E+10 1.00E+12
 1.00E+00 1.00E+01 1.00E+02
 7.00E-14 8.00E-14 9.00E-14
 1.00E-13 2.00E-13 3.00E-13
C-----------------------------------------------------------------------
C  ISEL  WAVELENGTH      TRANSITION            TYPE    METASTABLE  IMET NMET  IP
C  ----  ----------  ------------------------  -----  ------------ ---- ----  --
C   1.      61.1     3(3)1( 4.0)- 1(1)0( 0.0)  EXCIT   1(1)0( 0.0)   1     2
C   2.      61.1     3(3)1( 4.0)- 1(1)0( 0.0)  EXCIT   2(3)0( 1.0)   2     2
C   3.      61.1     3(3)1( 4.0)- 1(1)0( 0.0)  RECOM  SEE WJD91#B             +1
C   4. Any downloaded OPEN-ADAS file is made available for the
"""


def _file(tmp_path, text, name="pec.dat"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_pec40_keys_units_and_wavelengths(tmp_path):
    rates, wavelengths = read_adf15(_file(tmp_path, PEC40), "ic")

    assert rates["recombination"] == {}
    assert list(rates["excitation"]) == [("ic:487", "ic:1"), ("ic:487", "ic:2")]
    first = rates["excitation"][("ic:487", "ic:1")]
    np.testing.assert_allclose(first["ne"], [1e16, 1e18])  # m^-3
    np.testing.assert_allclose(first["te"], [1, 10, 100])  # eV
    np.testing.assert_allclose(first["rate"], [[1e-18, 2e-18, 3e-18], [4e-18, 5e-18, 6e-18]])
    # Fortran D exponents
    np.testing.assert_allclose(rates["excitation"][("ic:487", "ic:2")]["rate"][0, 0], 1e-19)
    assert wavelengths == {("ic:487", "ic:1"): pytest.approx(12.031),
                           ("ic:487", "ic:2"): pytest.approx(13.3398)}


def test_pec93_keeps_ground_metastable_and_recombination(tmp_path):
    rates, wavelengths = read_adf15(_file(tmp_path, PEC93), "llr")

    transition = ("llr:3", "llr:1")
    assert list(rates["excitation"]) == [transition]
    # the metastable-2 block (all 9e-12) is skipped
    np.testing.assert_allclose(rates["excitation"][transition]["rate"][0], [1e-18, 2e-18, 3e-18])
    np.testing.assert_allclose(rates["recombination"][transition]["rate"][1], [1e-19, 2e-19, 3e-19])
    assert wavelengths == {transition: pytest.approx(6.11)}


def test_duplicate_transition_raises(tmp_path):
    text = PEC40.replace("487(2)3(  3.5)-  2(2)3", "487(2)3(  3.5)-  1(2)3")
    with pytest.raises(ValueError, match="two excitation blocks for transition 487 -> 1"):
        read_adf15(_file(tmp_path, text), "ic")


def test_block_missing_from_transition_table_raises(tmp_path):
    text = PEC40.replace("C      2   133.398", "C      9   133.398")
    with pytest.raises(ValueError, match="block 2 is missing from the transition table"):
        read_adf15(_file(tmp_path, text), "ic")


def test_unknown_block_type_raises(tmp_path):
    text = PEC93.replace("TYPE = RECOM", "TYPE = CHEXC")
    with pytest.raises(ValueError, match="block type 'chexc' is not supported"):
        read_adf15(_file(tmp_path, text), "llr")


def test_truncated_file_raises(tmp_path):
    text = PEC40.replace("    2    /W : 7", "    3    /W : 7")
    with pytest.raises(ValueError, match="file ends after 2 of 3 blocks"):
        read_adf15(_file(tmp_path, text), "ic")


def test_skipped_metastable_blocks_warn(tmp_path):
    with pytest.warns(UserWarning, match="skipped 1 blocks driven from metastables"):
        read_adf15(_file(tmp_path, PEC93), "llr")


def test_wavelength_differs_from_transition_table_raises(tmp_path):
    text = PEC40.replace("C      2   133.398", "C      2   134.398")
    with pytest.raises(ValueError, match="block 2 is at 133.398 A, its transition table row at 134.398 A"):
        read_adf15(_file(tmp_path, text), "ic")


def test_partition_resolved_block_raises(tmp_path):
    text = PEC40.replace("ispb = t/isel =    2", "ispb = 2/isel =    2")
    with pytest.raises(ValueError, match="partition-resolved block \\(ispb = 2\\)"):
        read_adf15(_file(tmp_path, text), "ic")


def test_damaged_download_raises_with_path(tmp_path):
    path = _file(tmp_path, "<html><body>Not found</body></html>\n")
    with pytest.raises(ValueError, match=f"{path}: not a readable ADF15 file"):
        read_adf15(path, "ic")

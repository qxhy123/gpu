from gpusim.core.regfile import bank_of, operand_extra_cycles

def test_bank_assignment_default_4_banks():
    assert bank_of("r0") == 0
    assert bank_of("r1") == 1
    assert bank_of("r4") == 0
    assert bank_of("f5") == 1

def test_no_extra_cycles_when_banks_distinct():
    assert operand_extra_cycles(["r0", "r1", "r2"]) == 0

def test_two_sources_same_bank_one_extra_cycle():
    assert operand_extra_cycles(["r0", "r4"]) == 1

def test_three_sources_all_same_bank_two_extra():
    assert operand_extra_cycles(["r0", "r4", "r8"]) == 2

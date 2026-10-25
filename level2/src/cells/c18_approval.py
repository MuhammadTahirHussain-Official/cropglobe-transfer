# ======================================================================================
# INSTRUCTOR APPROVAL (fill this cell in ONLY after the instructor has signed the decisions).
#   1. Run every cell above this one.  The plan cell printed the DESIGN HASH below and saved it to tables/design_hash.txt.
#   2. Send the instructor: the plan table, tables/feasibility_grid.csv, and this hash.
#   3. After he/she signs, paste the hash here and set INSTRUCTOR_APPROVED = True.  Run this cell, then the run cell.
#   If anything in the design changes later (a setting, a budget, a seed, the data), the hash changes and the run cell refuses to start.
# ======================================================================================
INSTRUCTOR_APPROVED = False
APPROVED_DESIGN_SHA256 = ''                      # <- paste the 64-character hash that the instructor approved
CURRENT = design_hash()
print('DESIGN HASH of the plan that would run now:\n  ', CURRENT)
print('approval switch:', INSTRUCTOR_APPROVED, '| hash pasted:', bool(APPROVED_DESIGN_SHA256), '| matches the current plan:', APPROVED_DESIGN_SHA256 == CURRENT)
print('=> the run cell will', 'START' if (INSTRUCTOR_APPROVED and APPROVED_DESIGN_SHA256 == CURRENT) else 'REFUSE to start (this is intended until the instructor has signed)')

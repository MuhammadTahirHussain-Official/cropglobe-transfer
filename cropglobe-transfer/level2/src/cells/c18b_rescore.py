# ======================================================================================
# RE-SCORE ALL USA ROWS WITH THE CLEAN FIXED-CLASS RULE  -- L2-D11, requested by the instructor after seeds 0-3 were seen.  NO retraining: it reads the SAVED predictions of each run.
#   rule: score corn, soybeans, wheat, other.  TRUE rice / sugarcane / cotton samples are removed from the USA test set (the paper does the same for sugarcane and cotton in FRA->USA, Sec. 4.4;
#         extending it to rice is the instructor's decision); predictions of rice / sugarcane / cotton are counted as 'other'.  The score no longer depends on which regions were in the training mix.
#   * covers EVERY seed, rows still on the original coverage-conditional rule and rows already on L2-D9; run it ONCE before the run cell; safe to run again (finished rows are skipped)
#   * it first checks that the saved predictions reproduce the numbers already in results.csv (if not: STOP, nothing is written)
#   * the original results.csv is kept byte-for-byte as results.csv.before_rescore_<time> (SHA-256 in rescore_log.json); earlier scores stay in main_f1_before_fixed4 / main_f1_before_clean
#   * the run cell refuses to continue while any finished USA row has another rule
# ======================================================================================
_res_path = os.path.join(OUT_DIR, 'results.csv')
if not os.path.exists(_res_path): print('no results.csv yet -> nothing to re-score (new runs are scored with the clean rule from the start).')
else:
    RESCORE_REPORT = rescore_usa_fixed4(OUT_DIR)
    if len(RESCORE_REPORT):
        R_ = RESCORE_REPORT.copy(); pd.set_option('display.width', 200)
        print(f'\nUSA rows re-scored: {len(R_)} (seeds {sorted(R_.seed.unique().tolist())}) | mean main F1 before {100 * R_.old.mean():.2f} -> after {100 * R_.new.mean():.2f}')
        print('by the rule the row had before:'); print((100 * R_.groupby('old_rule')[['old', 'v1', 'new']].mean()).round(2).join(R_.groupby('old_rule').size().rename('n')).to_string())
        print(f'per run: true samples removed (rice/sugarcane/cotton) mean {R_.n_true_dropped.mean():.0f}, predictions moved to other mean {R_.n_pred_reassigned.mean():.0f}, samples scored mean {R_.n_scored.mean():.0f}')
        R_.drop(columns=['i']).to_csv(os.path.join(OUT_DIR, 'tables', 'rescore_fixed4_clean_rows.csv'), index=False)
    else: print('all finished USA rows already use the clean fixed-class rule.')

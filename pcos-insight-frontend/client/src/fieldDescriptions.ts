// Display guidance only. Internal feature names and submitted values are unchanged.
const descriptions: Record<string, string> = {
  'Age (yrs)': 'Your age in completed years.',
  'BMI': 'Body mass index: weight in kilograms divided by height in metres squared.',
  'Waist:Hip Ratio': 'Waist circumference divided by hip circumference, using the same units.',
  'Cycle(R/I)': 'Regular: periods follow a consistent pattern; irregular: timing varies or periods are missed.',
  'Cycle length(days)': 'Number of days bleeding lasts during a period—not the number of days between periods.',
  'Weight gain(Y/N)': 'Choose Yes if weight gain is reported for this assessment; otherwise choose No.',
  'hair growth(Y/N)': 'Choose Yes for excess facial or body hair; otherwise choose No.',
  'Skin darkening (Y/N)': 'Choose Yes if skin darkening is reported; otherwise choose No.',
  'Hair loss(Y/N)': 'Choose Yes if hair thinning or hair loss is reported; otherwise choose No.',
  'Pimples(Y/N)': 'Choose Yes if pimples or acne are present; otherwise choose No.',
  'Fast food (Y/N)': 'Choose Yes if fast-food intake is reported for this assessment; otherwise choose No.',
  'Reg.Exercise(Y/N)': 'Choose Yes if regular exercise is reported for this assessment; otherwise choose No.',
  'Vit D3 (ng/mL)': 'Vitamin D3 blood-test result in ng/mL, copied from your lab report.',
  'FSH(mIU/mL)': 'Follicle-stimulating hormone (FSH) blood-test result in mIU/mL.',
  'LH(mIU/mL)': 'Luteinizing hormone (LH) blood-test result in mIU/mL.',
  'TSH (mIU/L)': 'Thyroid-stimulating hormone (TSH) blood-test result in mIU/L.',
  'AMH(ng/mL)': 'Anti-Müllerian hormone (AMH) blood-test result in ng/mL.',
  'PRL(ng/mL)': 'Prolactin blood-test result in ng/mL, copied from your lab report.',
  'PRG(ng/mL)': 'Progesterone blood-test result in ng/mL, copied from your lab report.',
  'Follicle No. (L)': 'Number of follicles in the left ovary, as recorded on your ultrasound report.',
  'Follicle No. (R)': 'Number of follicles in the right ovary, as recorded on your ultrasound report.',
  'Avg. F size (L) (mm)': 'Average follicle size in the left ovary, in millimetres, from your ultrasound report.',
  'Avg. F size (R) (mm)': 'Average follicle size in the right ovary, in millimetres, from your ultrasound report.',
  'Endometrium (mm)': 'Thickness of the uterine lining in millimetres, from your ultrasound report.',
};

export function describeField(name: string): string {
  return descriptions[name.trim()] ?? 'Enter the value recorded for this assessment in the displayed units.';
}

export function helpId(name: string): string {
  return `input-help-${name.trim().replace(/[^a-zA-Z0-9]+/g, '-')}`;
}

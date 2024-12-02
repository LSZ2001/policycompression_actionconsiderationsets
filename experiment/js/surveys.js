var demographic_age = {
  type: 'survey-text',
  questions: [
  {prompt: 'What is your age?', required: true},
  ]}

var demographic_gender = {
    type: 'survey-multi-choice',
    questions: [
    {prompt: 'What is your gender?', options: ['M','F','non-binary','prefer not to say'], name: 'gender',  required: true}
    ]}

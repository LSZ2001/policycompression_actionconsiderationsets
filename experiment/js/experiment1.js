$(document).ready(function(){

  /* Save data to CSV */
function saveData(name, data) {
  var xhr = new XMLHttpRequest();
    xhr.open('POST', 'write_data.php'); // 'write_data.php' is the path to the php file described above.
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.send(JSON.stringify({
      filename: name,
      filedata: data
    }));
}

function downloadCSV(csv, filename) {
  // Retrive csv file from task
  var csvFile = new Blob( [csv], {type: "text/csv"});
  // Download link
  var downloadlink = document.createElement("a");
  // Download link download
  downloadlink.download = filename;
  downloadlink.href = window.URL.createObjectURL(csvFile);
  downloadlink.style.display = 'none';
  document.body.appendChild(downloadlink)
  downloadlink.click()
}

  var timeline = [];
  var finish = false;

  
  /* Obtain consent */
  var consent_block = {
    type: 'external-html',
    url: "consent_exp1.html",
    cont_btn: "start",
    check_fn: check_consent
  };

  /* Full screen */
  var fullscreen = {
    type: 'fullscreen',
    fullscreen_mode: true
  }

timeline.push(consent_block);
timeline.push(fullscreen);


var iti_shared = 500; //1000;
var reward_points = [1,0.2,-0.18,-1]; // Response correct, Safe choice, Response incorrect, no response.

// Stimuli action pairing
var stimuli = [
  'img/frac1.png', 'img/frac2.png', 'img/frac3.png', 'img/frac4.png', 'img/frac5.png', 'img/frac6.png', 'img/frac7.png', 'img/frac8.png', 'img/frac9.png', 'img/frac10.png',
  'img/frac11.png', 'img/frac12.png', 'img/frac13.png', 'img/frac14.png', 'img/frac15.png', 'img/frac16.png', 'img/frac17.png', 'img/frac18.png', 'img/frac19.png', 'img/frac20.png',
  'img/frac21.png', 'img/frac22.png', 'img/frac23.png'
];
stimuli = jsPsych.randomization.repeat(stimuli,1,false);
// For Experiment 2: Make one action the optimal action for 2 states.
  function replaceRandomEntry(array) {
      // Step 1: Randomly select an index to replace
      const indexToReplace = Math.floor(Math.random() * array.length);
      // Step 2: Create a copy of the array and remove the element at the selected index
      const remainingValues = array.filter((_, index) => index !== indexToReplace);
      // Step 3: Randomly select a new value from the remaining values
      const newValueIndex = Math.floor(Math.random() * remainingValues.length);
      const newValue = remainingValues[newValueIndex];
      // Step 4: Replace the original value with the new value
      const newArray = [...array];
      newArray[indexToReplace] = newValue;
      return newArray;
  }
var actions = [49,50,51,52,53,54];
var trial_choices = ['1', '2', '3', '4', '5', '6', 'E'];
var correct_actions = actions;
var safe_action = 69; // The key "E" on the keyboard.

// Period of 8. 
var stimuli_trials = [
  { stimuli_trials: stimuli[0], data_trials:{state: 1, test_part:'trials', correct_response:correct_actions[0]}},
  { stimuli_trials: stimuli[1], data_trials:{state: 2, test_part:'trials', correct_response:correct_actions[1]}},
  { stimuli_trials: stimuli[2], data_trials:{state: 3, test_part:'trials', correct_response:correct_actions[2]}},
  { stimuli_trials: stimuli[3], data_trials:{state: 4, test_part:'trials', correct_response:correct_actions[3]}},
  { stimuli_trials: stimuli[4], data_trials:{state: 5, test_part:'trials', correct_response:correct_actions[4]}},
  { stimuli_trials: stimuli[5], data_trials:{state: 6, test_part:'trials', correct_response:correct_actions[5]}},
];
const multiplyArray = (arr, length) =>  Array.from({ length }, () => arr).flat()
var stimuli_trials_block;

var k = 0;
images_html = '<p>';
while (k < actions.length) {
  images_html = images_html + ' <img src= '+stimuli_trials[k]['stimuli_trials']+' style="width:150px; height:150px;">';
  k++;
}
images_html += '</p>';

var images_html_correctresp = '<div style="text-align:center;">'; // Start the div container for center alignment of images
var number_html = '<div style="display:flex; justify-content:center; margin-top:10px;">'; // Start flexbox container for numbers and header
var k = 0;
while (k < actions.length) {
  // Add the image with padding between images
  images_html_correctresp += '<div style="display:inline-block; text-align:center; padding: 0 10px;">'; // Add horizontal padding of 10px
  images_html_correctresp += '<img src="'+stimuli_trials[k]['stimuli_trials']+'" style="width:150px; height:150px;"><br>';
  images_html_correctresp += '</div>';
  // Add the correct response from stimuli_trials[k]['data_trials']['correct_response']
  number_html+= '<div style="display:inline-block; width:150px; text-align:center; padding: 0 10px; font-size:20px"> <b>'+(stimuli_trials[k]['data_trials']['correct_response']-48)+'</b></div>';
  k++;
}
images_html_correctresp += '</div>'; // Close the image row div
// Combine both rows: one with images and one with correct responses and the header
images_html_correctresp += number_html;




// Instantiate variables
var num_blocks = 7;  // 4 train, 3 test
var totalPracticeBlocks = 4;
var num_correctresponsegiven_blocks = 1;
var time_limits_unique = [2000,1000,500];
var time_limits = jsPsych.randomization.repeat(time_limits_unique, 1, false) //  time_limit for response: 500, 2000, 5000 milliseconds
time_limits = [2000].concat(time_limits_unique).concat(time_limits)
var num_trials = [60,60,60,60,96,96,96]; // 20
//var num_trials = [6,6,6,6,12,12,12]; // 20
var provideInstructions = [false,false,false,false,false,false,false];
var provideCorrectFeedback = [true,false,false,false,false,false,false];


var green = "#04b104";
var pink = "#ffc0cb";
var light_green = "#b5fbb5";

// Let subjects check if their keyboard presses can be recorded.
function input_check(key_choice){
  var input_check_page = {
    type: 'html-keyboard-response',
    stimulus: function(){
      if(key_choice=="E"){
        return '<p>To check if your keyboard presses can be successfully recorded, <br> please press the letter key <b>"'+key_choice+'"</b> to proceed.</p> If you can\'t proceed, you won\'t be able to complete this study.'
      } else {
        return '<p>To check if your keyboard presses can be successfully recorded, <br> please press the number key <b>"'+key_choice+'"</b> to proceed.</p> If you can\'t proceed, you won\'t be able to complete this study.'
      }
    },
    choices: [key_choice],  
  };
  return input_check_page
}
for (var i=0; i<trial_choices.length; i++) {
  var input_check_page = input_check(trial_choices[i])
  timeline.push(input_check_page)
}



var instructions_exp = {
    type: 'instructions',
    pages: [
      // Welcome (page 1)
      '<p class="center-content"><b>Welcome to our study! </p>' +
      '<p class="center-content">In this study, you will earn <b style="color:'+green+'">$6</b> plus a bonus of <b style="color:'+green+'">$1-2</b> depending on your performance!</b></p>' +
      '<p class="center-content">This task should take you about <b>25 minutes</b> in total. </p>' +
      '<p class="center-content">Press "Next" to view the instructions.</p>',
      // Instructions (page 2)
      '<p> In this study, you will press number keys on your keyboard: <b>"1","2","3","4","5","6"</b>, and the letter key <b>"E"</b>, for different images on the screen.</p>' +
      '<p> On each trial, you will see one image such as these:</p>' +
      images_html,

      '<p> For each image, pressing some number key would always give you <b style="color:'+green+'">+'+reward_points[0]+'</b> <b>reward point</b>.'+
      '<br> Pressing all other number keys would always incur a loss of <b style="color:'+pink+'">'+reward_points[2]+'</b> <b>reward points</b>.'+
      '<p> However, for any image, pressing <b>"E"</b> would always give you <b style="color:'+light_green+'">+'+reward_points[1]+'</b> <b>reward points</b>.',

      //Instructions (page 6)
      '<p> If your key press yielded <b style="color:'+green+'">+'+reward_points[0]+'</b> reward point, a <b style="color:'+green+'">green border</b> will appear around the image, like this: </p>' +
      '<p> <img src= '+stimuli[0]+' style="border:14px solid '+green+'; width:150px; height:150px;"></p>'+
      '<p> If your key press yielded <b style="color:'+pink+'">'+reward_points[2]+'</b> reward points, a <b style="color:'+pink+'">pink border</b> will appear around the image, like this:</p>'+
      '<p> <img src= '+stimuli[0]+' style="border:14px solid '+pink+'; width:150px; height:150px;"> </p>'+
      '<p> If your key press yielded <b style="color:'+light_green+'">+'+reward_points[1]+'</b> reward points, a <b style="color:'+light_green+'">light green border</b> will appear around the image, like this:</p>'+
      '<p> <img src= '+stimuli[0]+' style="border:14px solid '+light_green+'; width:150px; height:150px;"> </p>',
  
      '<p> Your <b>goal</b> in this task is to <b>get as many reward points as possible.</b></p>'+
      '<p> The experiment has <b>3 blocks</b>, each containing <b>'+num_trials[num_trials.length - 1]+' trials.</b>' + 
      '<p> For each block, there is a <b style="color:red">maximum time limit</b> you can spend per trial. <br> You must respond with a key press before then. Otherwise, you lose <b style="color:red">'+reward_points[3]+'</b> reward points.' + 
      '<p> Overall, <b>it is most rewarding to <b style="color:'+green+'">press a correct number key (+'+reward_points[0]+')</b>, followed by <b style="color:'+light_green+'">pressing "E" (+'+reward_points[1]+')</b>, <br> followed by <b style="color:'+pink+'">pressing an incorrect number key ('+reward_points[2]+')</b>, followed by <b style="color:red">not responding within the time limit ('+reward_points[3]+')</b>.</b>'+
      '<p>Your bonus pay $$ depends on the reward you get in each block (can be negative), relative to other participants in that block.'+
      '<br> Your base pay is always ensured, and the bonus pay itself is always positive.',

      '<p> Now let us talk more about <b style="color:red">time limits</b>, which differs across blocks.</p>'+
      '<p> The time limit of a block can be either <b>0.5s, 1s, or 2s</b>. We will tell you before starting each block.</p>'+
      '<p> Therefore, think about how to get as many rewards as possible!</p>'+
      "<p> Don't worry: <b>your submission will be approved</b> no matter what <b>strategy</b> you use, no matter how much <b>reward</b> you earn.</p>"+
      '<p> However, there is one exception: if you fail to respond within the <b style="color:red">time limit</b> for more than <b>50 trials across all blocks</b>, <br> We will <b>reject</b> your submission based on insufficient engagement.</p> <p> Hence, please try your best to <b>respond to every trial in time</b>, all the way until the end!</p>',

      '<p style="color:black"> Click "Next" to go through <b>'+totalPracticeBlocks+' practice blocks</b> with different <b style="color:red">time limits</b> but fewer trials. </p> <p>Practice blocks have no bearing on your bonus payment.</p>',
  
    ],
    show_clickable_nav: true,
    allow_backward: true,
    show_page_number: true
  };
  
  console.log('instructions loaded')

timeline.push(instructions_exp);



// TRIALS BLOCK // 
var currentBlock = 0;

      function timerDisplay() {
        return '<p style="position:fixed; top:27%; left:44.5%"> Reward so far: <b style="color:'+green+'">'+Math.round(10*pointsInBlock)/10+'</b></p>'
      }


  var pointsInBlock = 0;

  for (i = 0; i < num_blocks; i++) {
    var total_rt = 0;
    var num_trials_perblock = num_trials[i];
    var time_limit = time_limits[i];
    var stimuli_trials_block = jsPsych.randomization.repeat(stimuli_trials,Math.round(num_trials_perblock/stimuli_trials.length),false);
    currentBlock++;

    var provide_instructions_block = provideInstructions[i];
    var provide_correct_feedback_block = provideCorrectFeedback[i];

    // Immediately-Invoked Function Expression (IIFE)
    (function(blockNumber, blockInterval, stimuliTrialsBlock, provide_instructions, provide_correct_feedback) {

      var prestart_block = {
        type: 'html-button-response',
        stimulus: function() {
          if (blockNumber>totalPracticeBlocks) {
            if(blockNumber==(totalPracticeBlocks+1)){ // First proper block
                var text_display = '<p class="center-content">Congratuations! You have completed all practice blocks.'+
                '<br> You will now proceed to the <b>3 blocks</b> that determine your bonus payment.<p>'+
                '<p class="center-content">As a refresher, here are the image-key mappings that will give you <b style="color:'+green+'">+'+reward_points[0]+'</b> reward point<b>:</p>'+
                images_html_correctresp;
                return [text_display];
            } else {
                return [''];
            }

          } else {
            if(blockNumber==num_correctresponsegiven_blocks){
              var text_display = '<p class="center-content">In <b>Practice Block 1</b>, after you press a key for an image,'+
              '<br> We will remind you on the "best" number key for that image, which would have yielded <b style="color:'+green+'">+'+reward_points[0]+'</b> reward.<p>'+
              '<p class="center-content">Here are the image-key mappings that will give you <b style="color:'+green+'">+'+reward_points[0]+'</b> reward point:</p>'+
              '<div style="text-align:center; margin-bottom: 20px;">' + images_html_correctresp + '</div>' + // Wrap images_html in a centered div+
              '<p><br> Using this practice block, please learn and practice these image-key mappings to the best of your ability.</p>';
              return [text_display];

            } else if(blockNumber==(num_correctresponsegiven_blocks+1)) {
              var text_display = '<p class="center-content">In <b>Practice Blocks 2-4</b>, we won\'t remind you on the "best" number key anymore.</p>'+
              '<p> As a refresher:'+
              '<br> You are free to press any key among <b>"1","2","3","4","5","6","E"</b>, within the block\'s <b style="color:red">time limit</b>.'+
              '<br> For each image, pressing some number key would always give you <b style="color:'+green+'">+'+reward_points[0]+'</b> <b>reward point</b>.'+
              '<br> Pressing all other number keys would always incur a loss of <b style="color:'+pink+'">'+reward_points[2]+'</b> <b>reward points</b>.'+
              '<br> However, for any image, pressing <b>"E"</b> would always give you <b style="color:'+light_green+'">+'+reward_points[1]+'</b> <b>reward points</b>.'+
              '<br> If you do not respond within the time limit, you lose <b style="color:red">'+reward_points[3]+'</b> <b>reward points</b>.'+
              '<p> Please use these practice blocks as a chance to develop your strategy for different <b style="color:red">time limits</b>.'+
                            images_html_correctresp;

              return [text_display];
              
            } else {
            return [''];
            }
          }
        },
        choices: ['Next'],
        on_start: function() {
        },
        on_finish: function() {
            pointsInBlock = 0;
            blockStartTime = Date.now(); // Start timing the block
        },
      };
      if(blockNumber==num_correctresponsegiven_blocks || blockNumber==num_correctresponsegiven_blocks+1 || blockNumber==totalPracticeBlocks+1){
        timeline.push(prestart_block)
      }

      var start_block = {
        type: 'html-button-response',
        stimulus: function() {
          if (blockNumber>totalPracticeBlocks) {
            return ['<p class="center-content">In <b>Block ' + (blockNumber - totalPracticeBlocks) + '</b> containing '+num_trials[blockNumber-1]+' trials,</p><p>You must respond to a trial within <b style="color:red">' + (Math.round(blockInterval / 100) / 10).toString() + ' sec</b>.</p>'];
          } else {
            return ['<p class="center-content">In <b>Practice Block ' + (blockNumber) +'</b> containing '+num_trials[blockNumber-1]+' trials,</p><p>You must respond to a trial within <b style="color:red">' + (Math.round(blockInterval / 100) / 10).toString() + ' sec</b>.</p>'];
          }
        },
        choices: ['Next'],
        on_start: function() {
        },
        on_finish: function() {
            pointsInBlock = 0;
            blockStartTime = Date.now(); // Start timing the block
        },
      };

      timeline.push(start_block);

      var trial = {
        type: 'image-keyboard-response',
        stimulus: jsPsych.timelineVariable('stimuli_trials'),
        timeline_variables: stimuli_trials,
        choices: function(){
          if(!provide_instructions){
            return trial_choices
          } else{
            var correct_action = jsPsych.timelineVariable('data_trials')().correct_response; 
            return [correct_action]
          }},
        data: jsPsych.timelineVariable('data_trials'),
        prompt: function() { 
          if(!provide_instructions){
            return timerDisplay(); 
          } else {
            var correct_action = jsPsych.timelineVariable('data_trials')().correct_response; 
            return '<p style="position:fixed; top:62%; left:45%;font-size:25px">Best key: <b>'+(correct_action-48)+'</b></p>';
          }
        },
        trial_duration: time_limit,
        on_finish: function(data) {
            trial_node_id = jsPsych.currentTimelineNodeID();
            if(Number(data.key_press[0]==safe_action)) {
              data.correct = reward_points[1];
              data.rewarded = reward_points[1];
            } else if(Number(data.key_press == data.correct_response)){
              data.correct = 1;
              data.rewarded = reward_points[0];
            } else if (data.key_press.length==0) {
              data.correct = reward_points[3];
              data.rewarded = reward_points[3];
            } else {
              data.correct = 0;
              data.rewarded = reward_points[2];
            }
            rt = data.rt;
            total_rt = +total_rt + +rt;
            pointsInBlock = pointsInBlock + data.rewarded;
            console.log(data.key_press)
            console.log(data.rewarded)
        }
      };

      var feedback = {
        type: 'html-keyboard-response',
        stimulus: function(){
          var prev_trial = jsPsych.data.getDataByTimelineNode(trial_node_id);
          var feedback_img = prev_trial.select('stimulus').values[0];
          var feedback = prev_trial.select('key_press').values[0];
          if (prev_trial.select('rewarded').values[0]==reward_points[0]){
            return  '<img src="' + feedback_img + '" style="border:14px solid '+green+';">';
          } else if (prev_trial.select('rewarded').values[0]==reward_points[1]){
            return '<img src="' + feedback_img + '" style="border:14px solid '+light_green+';">';
          } else if (prev_trial.select('rewarded').values[0]==reward_points[3]){
            return '<img src="' + feedback_img + '" style="border:14px solid red;">';
          } else {
            return '<img src="' + feedback_img + '" style="border:14px solid '+pink+';">';
          }
        },
        trial_duration: function(){
          if(provide_correct_feedback){
            return 1000
          } else{
            return 500}
          },
        prompt: function() { 
          if(provide_instructions){
            var correct_action = jsPsych.timelineVariable('data_trials')().correct_response; 
            return '<p style="position:fixed; top:62%; left:45%; font-size:25px">Best key: <b>'+(correct_action-48)+'</b></p>';
          }else if(provide_correct_feedback){
            var correct_action = jsPsych.timelineVariable('data_trials')().correct_response; 
            return timerDisplay()+'<p style="position:fixed; top:62%; left:45%; font-size:25px">Best key: <b>'+(correct_action-48)+'</b></p>';
          }else {
              return timerDisplay(); 
          }
        },
        choices: jsPsych.NO_KEYS,
        on_finish: function(data){
          if(provide_correct_feedback){
            total_rt = +total_rt+ +1000;
            data.feedback_duration = 1000;
          } else {
            total_rt = +total_rt + +500;
            data.feedback_duration = 500;
          }
        }
      };

      var fixation = {
        type: 'html-keyboard-response',
        stimulus: function() {
          var prev_trial = jsPsych.data.getDataByTimelineNode(trial_node_id);
          var prev_reward = prev_trial.select('rewarded').values[0];

          if (prev_trial.select('rewarded').values[0]==reward_points[0]){
            return  '<div style="font-size:60px; color:'+green+'">+'+prev_reward+'</div>';
          } else if (prev_trial.select('rewarded').values[0]==reward_points[1]){
            return '<div style="font-size:60px; color:'+light_green+'">+'+prev_reward+'</div>';
          } else if (prev_trial.select('rewarded').values[0]==reward_points[3]){
            return'<div style="font-size:60px; color:red">'+prev_reward+'</div>';
          } else {
            return'<div style="font-size:60px; color:'+pink+'">'+prev_reward+'</div>';
          }
        },
        prompt: function() {
          var prev_trial = jsPsych.data.getDataByTimelineNode(trial_node_id);
          if((!provide_instructions)&&(!provide_correct_feedback)){
            if(prev_trial.select('rewarded').values[0]!=reward_points[3]){
              return timerDisplay();
            } else {
              return timerDisplay() + '<p><b style="color:red; font-size:30px;"> Too slow! </b></p>'
            }
          }else if(provide_instructions){
            if(prev_trial.select('rewarded').values[0]!=reward_points[3]){
              var correct_action = jsPsych.timelineVariable('data_trials')().correct_response; 
              return '<p style="position:fixed; top:62%; left:45%; font-size:25px">Best key: <b>'+(correct_action-48)+'</b></p>';
            } else {
              return '<p><b style="color:red; font-size:30px;"> Too slow! </b></p>';
            }
          }else{
            var correct_action = jsPsych.timelineVariable('data_trials')().correct_response; 
            if(prev_trial.select('rewarded').values[0]!=reward_points[3]){
              return timerDisplay() + '<p style="position:fixed; top:62%; left:45%; font-size:25px">Best key: <b>'+(correct_action-48)+'</b></p>';
            } else {
              return timerDisplay() + '<p style="position:fixed; top:62%; left:45%; font-size:25px">Best key: <b>'+(correct_action-48)+'</b></p>'+
              '<p><b style="color:red; font-size:30px;"> Too slow! </b></p>'
            }
          }
        },
        choices: jsPsych.NO_KEYS,
        trial_duration: iti_shared,
        data: {trial_iti: iti_shared, trial_deadline: time_limit},
        on_finish: function(data){
          total_rt = +total_rt + +jsPsych.data.get().last(1).select('trial_iti').values;
          data.total_rt = total_rt;
          data.block = blockNumber;
          data.correct_actions = correct_actions;
          data.is_practice = Number(blockNumber<=totalPracticeBlocks);
          data.provide_corraction_feedback = Number(provide_correct_feedback);
        }
      }
      var block = {
        timeline: [trial, feedback, fixation],
        timeline_variables: stimuliTrialsBlock,
        randomize_order: true,
        loop_function: function(data){
          return false;

        }
      };

      var end_block = {
        type: 'html-button-response',
        stimulus: function(){
          if(blockNumber==totalPracticeBlocks){
            return ['<p class="center-content">You have completed all practice blocks!</p>' 
            + '<p> Your reward points gained for the block was <b  style="color:#32CD32">' + Math.round(100*pointsInBlock)/100 + ' points</b>.' 
            + '<p> &nbsp; </p>'
            + '<p class="center-content"> You will now proceed to the <b>proper experiment</b> containing '+(num_blocks-totalPracticeBlocks)+' blocks, where your <b style="color:#32CD32">reward points</b> determine your bonus payment.</p>' 
            + '<p class="center-content">  Click "Next" to begin the experiment.</p>']
          } else if (blockNumber<num_blocks){
          return ['<p class="center-content">You have completed a block!</p> <p> Your reward points gained for the block was <b style="color:#32CD32">' + Math.round(100*pointsInBlock)/100 + ' points</b>.</p> <p> &nbsp; </p> <p>Take a break if you would like and then press "Next" to continue to the next block.</p><p><b>Note: </b> for each image, the reward associated with each key press remains the same across blocks.</p>',]
          } else {
          return ['<p class="center-content">You have completed a block!</p> <p> Your reward points gained for the block was <b  style="color:#32CD32">' + Math.round(100*pointsInBlock)/100 + ' points</b>.</p>']

          }
        },
        choices: ["Next"],
      };

      timeline.push(block);
      timeline.push(end_block);

    })(currentBlock, time_limit, stimuli_trials_block, provide_instructions_block, provide_correct_feedback_block);

}// loop through ITIs


// Define redirect link for Qualtrics and add Turk variables
var turkInfo = jsPsych.turk.turkInfo();

// Add MTurk info to CSV
jsPsych.data.addProperties({
  assignmentID: turkInfo.assignmentId
});
jsPsych.data.addProperties({
  mturkID: turkInfo.workerId
});
jsPsych.data.addProperties({
  hitID: turkInfo.hitId
});


// End of all blocks //
var end_all_blocks = {
  type: 'html-button-response',
  stimulus: '<p class="center-content"> <b>And...that was the last block!</b></p>' +
  '<p class="center-content"> On the next pages, we will ask some demographic questions and save your data. Then the study is complete!</p>',
      choices: ['Next'],
};

var demographics = {
  timeline: [demographic_age, demographic_gender],
  repetitions: 1
};

var survey_strategy = {
  type: "survey-text",
  questions: [{prompt: 'What is your strategy across blocks with different time limits?', value: 'Strategy', rows: 10, required: true}],
};

var survey_feedback = {
  type: "survey-text",
  questions: [{prompt: 'Please let us know if you have any feedback for our experiment.', value: 'Feedback', rows: 10, required: false}],
};

// Save data //
var save_data = {
  type: "survey-text",
  questions: [{prompt: 'Please input your MTurk Worker ID so that we can pay you the appropriate bonus. Your ID will not be shared with anyone outside of our research team. Your data will now be saved.', value: 'Worker ID'}],
  on_finish: function(data) {
  var responses = JSON.parse(data.responses);
  var subject_id = responses.Q0;
  console.log(subject_id)
  saveData(turkInfo.workerId, jsPsych.data.get().csv());
  },
};

// End of experiment //
var end_experiment = {
  type: 'instructions',
  pages: [
      '<p class="center-content"> <b>Thank you for participating in our study!</b></p>' +
      '<p class="center-content"> Here is your completion code: <b style="color: red"> C2t8DTNoQBUm </b> </p>' + 
      '<p> &nbsp; </p>' +
      '<p class="center-content"> <b>Please wait on this page for 1 minute while your data saves.</b></p>'+
      '<p class="center-content"> Your bonus will be applied after your data has been processed and your HIT has been approved. </p>'+
      '<p class="center-content"> Please email shuzeliu@g.harvard.edu with any additional questions or concerns!</p>'
    ],
  show_clickable_nav: false,
  allow_backward: false,
  show_page_number: false,
  allow_keys:false,
};


timeline.push(end_all_blocks)
timeline.push(demographics)
timeline.push(survey_strategy)
timeline.push(survey_feedback)
timeline.push(save_data);
timeline.push(end_experiment);


function startExperiment(){
  jsPsych.init({
    timeline: timeline,
    show_progress_bar: true,
    auto_update_progress_bar: true,
  })
};


jsPsych.pluginAPI.preloadImages(stimuli, function () {startExperiment();});
console.log("Images preloaded.");

})

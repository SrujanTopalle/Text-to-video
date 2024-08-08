from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

def summarize_text(text, max_length=140, min_length=50, do_sample=False):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

text = """
At the top of many automation wish lists is a particularly time-consuming task: chores. 

The moonshot of many roboticists is cooking up the proper hardware and software combination so that a machine can learn “generalist” policies (the rules and strategies that guide robot behavior) that work everywhere, under all conditions. Realistically, though, if you have a home robot, you probably dont care much about it working for your neighbors. MIT Computer Science and Artificial Intelligence Laboratory (CSAIL) researchers decided, with that in mind, to attempt to find a solution to easily train robust robot policies for very specific environments.

“We aim for robots to perform exceptionally well under disturbances, distractions, varying lighting conditions, and changes in object poses, all within a single environment,” says Marcel Torne Villasevil, MIT CSAIL research assistant in the Improbable AI lab and lead author on a recent paper about the work. “We propose a method to create digital twins on the fly using the latest advances in computer vision. With just their phones, anyone can capture a digital replica of the real world, and the robots can train in a simulated environment much faster than the real world, thanks to GPU parallelization. Our approach eliminates the need for extensive reward engineering by leveraging a few real-world demonstrations to jump-start the training process.”

RialTo, of course, is a little more complicated than just a simple wave of a phone and (boom!) home bot at your service. It begins by using your device to scan the target environment using tools like NeRFStudio, ARCode, or Polycam. Once the scene is reconstructed, users can upload it to RialTo’s interface to make detailed adjustments, add necessary joints to the robots, and more.

The refined scene is exported and brought into the simulator. Here, the aim is to develop a policy based on real-world actions and observations, such as one for grabbing a cup on a counter. These real-world demonstrations are replicated in the simulation, providing some valuable data for reinforcement learning. “This helps in creating a strong policy that works well in both the simulation and the real world. An enhanced algorithm using reinforcement learning helps guide this process, to ensure the policy is effective when applied outside of the simulator,” says Torne.

Testing showed that RialTo created strong policies for a variety of tasks, whether in controlled lab settings or more unpredictable real-world environments, improving 67 percent over imitation learning with the same number of demonstrations. The tasks involved opening a toaster, placing a book on a shelf, putting a plate on a rack, placing a mug on a shelf, opening a drawer, and opening a cabinet. For each task, the researchers tested the system’s performance under three increasing levels of difficulty: randomizing object poses, adding visual distractors, and applying physical disturbances during task executions. When paired with real-world data, the system outperformed traditional imitation-learning methods, especially in situations with lots of visual distractions or physical disruptions.

“These experiments show that if we care about being very robust to one particular environment, the best idea is to leverage digital twins instead of trying to obtain robustness with large-scale data collection in diverse environments,” says Pulkit Agrawal, director of Improbable AI Lab, MIT electrical engineering and computer science (EECS) associate professor, MIT CSAIL principal investigator, and senior author on the work.

As far as limitations, RialTo currently takes three days to be fully trained. To speed this up, the team mentions improving the underlying algorithms and using foundation models. Training in simulation also has its limitations, and currently it’s difficult to do effortless sim-to-real transfer and simulate deformable objects or liquids.
"""

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

summary = summarize_text(text)

print(summary)

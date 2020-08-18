**I [jonny] have made changes to the source code on a new branch (todos, improved comments, and a few changes to function/variable names etc)**. Where possible I have edited as I go, but am hoping for your input on resolving the queries in the todos. The more important conceptual queries are listed below. **I have also included a preliminary sketch of the architecture I have in mind**, although this is currently limited by my understanding of how the flask stuff works, particularly around multithreading, since that will have a big influence on what a correct and best way to handle all this in python will be. My biggest hangup is that I cannot see any connection between app.py and camera_pi.py (and in particular the frames() iterator). I therefore cannot see how the Flask app stuff is meant to work when running live on a RPi with the PiCam.

------------------------------------ **Queries to Chas about the different modules ------------------------------------**

cpi.py

OpticalGater class that extends picamera.array.PiYUVAnalysis. It appears to maintain history, e.g. frame_history, period_history.

**For chas:** my opinion is frame_history is really crying out for a class container, or at least an array with named columns (or a list-of-dictionaries, though I'm less keen on that). At present it requires the developer to keep track of the meaning of each column, and the format not easily extensible.

**For chas:** load_data() - "Place holder function for loading data in emulator". What is the intention of this, and is there a clear idea of what an emulator would look like?

**For me:** this is a general question for Chas: how does he intend the emulator to work? I don't think I've fully followed his emulator stuff in "app". Might need to talk through that one together...

**For chas: IMPORTANT, BUG:** as per my comment in this module, I don't think barrier frames are implemented properly.

**For chas: IMPORTANT, BUG:** I think there is a bug associated with (lack of) numExtraRefFrames in initialise_internal_parameters(), as per my comment there.

**For chas: IMPORTANT, approaching a BUG**: there is the general issue that the code is not doing a great job of precise timing. It determines a delay time before sending the trigger, but then executes a bunch more code. Oh, and that delay time is then treated relative to "current_time", which is set *after* doing the phase-matching. This is going to reduce accuracy and precision, and also makes me even more uncomfortable in terms of future-proofing. I think it would be much better to pass around absolute times, not deltas.

More generally, I wish we were working with frame *objects* that have accompanying metadata (such as timestamp!). That would help solve the above issue about timings in a tidy manner, and also make it a lot easier to generalise to other timebases (such as camera timebase, rather than computer). Oh, and it would also help deal with gaps in the timebase (missing frames), which I think would be a useful thing to cope with, even if right now there might not be any concrete scenario in which frames would be dropped. [JT: although we have not settled on a solution, we seem to be in agreement that we probably need some sort of a way of connecting metadata with the frame data]

Remark **for chas:** currently when we schedule a trigger everything hangs, waiting for the appropriate time to arrive. This stalls camera and sync analysis. Don't you think we should really figure out a way of doing this asynchronously, without blocking (or come to a definite conclusion that blocking is the only reliable enough way to do it on the Pi).

**For me: where do the stages come in?** cli.py sets up the usb/serial link, but where is it used, by whom, and when? Should be clearer after a refactor. Either we need to conditionally interact with the stages, or we do it on a more distributed basis, in response to events. That's more how I do it in my code, and I like that approach.

determine_reference_period.py

**For me and chas:** here and elsewhere, I'm happy to help refactor into a class myself. This module definitely needs some sort of refactoring (see "WTF"/"Ugh" comments).

**For chas**: padding flag seems to be used fairly inconsistently (would affect the if test in establish_indices), and when False they *still* seem to include extra ref frames, just fewer of them. Weird.... In fact, do you ever *not *use it, or should we just remove the parameter?

**For chas**: **IMPORTANT:** there is a structural issue here where the period-determination code processes one big block of frame buffers all together, once. That is not a good solution when running live - it will be much better to make a new attempt to determine a period after each individual new frame is received, rather than doing a massive chunk of processing in the loop in establish_indices. I really think this needs a refactor. Otherwise, apart from anything else, we will end up acquiring significantly more frames than we need to before we can establish a period. There are also scenarios where this approach will fail, where it could succeed if it was allowed just to keep trying indefinitely, processing a rolling buffer of recent frames. (Minor related observation: once that is changed, and the loop is removed from this function, the "didn't find a period" error will not be "critical" when running live)

prospective_optical_gating.py

[No comments beyond the inline ones in the source code]

emulate.py

I haven't really read through this in detail yet, but I do remember that I didn't find it particularly helpful as an example to follow when I was trying to do my own offline analysis using this codebase. I need to go back and remind myself what the difficulties were that I encountered...

stage_control_functions.py

Looks pretty well commented and structured, but I would implement this as a class, so that other (sub-)classes could be substituted to control other stage types, etc. I will give this some more thought. **For me**: how is this used? Which object is the most sensible one to be in charge of all this?

app

For my benefit: base_camera.py runs a background thread which posts events whenever a frame is available. frames() uses "yield" to pass frame buffers to an iterator.

app.gen() is a "video streaming generator function" which in turn "yield"s frames (in what looks like a MIME-like format) to the caller. I am not clear how video_feed behaves, though.

Basic questions **for chas**:

- What is the overall function of app.py? Do I understand correctly that these decorated functions are called in response to messages from the client webpage? In response to those messages, they appear to both perform internal state-affecting actions (i.e. run the synchronization algorithms, in some cases), *and also* then return the actual HTTP source(*) that should now be displayed in the webpage. Am I understanding that correctly? What is the deal with the "session" object? Could really do with a comment at the top of the module to explain what's going on with all of this.                (*) Oh, or sometimes raw video images apparently, in video_feed()...

- base_camera.py needs a similar comment at the top of the module explaining what's going on - and in particular what the relationship is between a "client" and a "thread", explanation of the role of "ident" (which is a unique identifier associated with each thread), etc. Are these concurrent threads or [I suspect] sequential threads? Really needs an explanation of the thread architecture here, and the role of "yield" in terms how of the threads interact.

- **IMPORTANT:** the same questions apply to the Flask app. As per my comment in app.py, I am starting to suspect that this could be running concurrent threads. That needs careful thought and documenting, because it gets very complicated.

- More generally, *as far as I can see*, there doesn't seem to be to be a critical need to be using [sequential] threads here at all (though it doesn't do any harm). Out of interest, is this code just adapted from some online example? I suspect it makes a lot more sense in the case where there is *more than one* client involved (but I don't think that's really an issue for us in reality).

**Out of interest:** I find the code formatting [newline after "(", and ")" on a line of its own, sometimes doing this in rather superfluous contexts] really weird. Is this an established coding style, or just what Alex does!? It's certainly one way to keep to line-width limits, but not the one I would choose - and line limits are violated elsewhere! Extreme example:

                                        self.trigger_fluorescence_image_capture(

                                                                                                                        current_time + timeToWaitInSecs

                                                                                                                        )

Hmm... although I can see that, in the absence of curly brackets for statement blocks, like in C, it helps with readability (in terms of the indenting layout) in cases like this:

if (period != -1

and len(periods) >= (5 + (2 * settings["numExtraRefFrames"]))

and period > 6

and (len(periods) - 1 - settings["numExtraRefFrames"]) > 0

and (periods[len(periods) - 1 - settings["numExtraRefFrames"]]) > 6

                ):

                        periodToUse = periods[len(periods) - 1 - settings["numExtraRefFrames"]]

------------------------------------ **Architecture sketch ------------------------------------**

I am still working on this, but I think there are benefits to having things in their own (concurrent) threads, which in python effectively means they are separate *processes* that must pass messages to each other (or be spawned and return asynchronously). For interfacing with M^2 system, I want a Sync Analyzer process that is driven by [low-level loopback IP?] messages sent back and forth between it and a client. On the RPi, we would also have a Sync Controller process, which would probably be the existing Flask loop, to take care of interfacing with camera/pins hardware. This Sync Controller would be in asynchronous communication with Sync Analyzer, but could also potentially be receiving GUI input from a web client interface.

Messages from Controller to Analyzer would mostly be messages containing frame data, but with other messages to send config information, reset, etc.

Messages from Analyzer to Controller would mostly be timestamps at which to schedule triggers, but with other messages such as informing when sync lock is acquired.

(In fact, the LTU might well end up being yet another separate process of its own, because it takes such a long time to complete its calculations, but that's a secondary detail...)

This architecture is helpful in terms of interfacing with the M^2 system (and others) - the Sync Analyzer process would just be in communication with their own software instead of the RPi Sync Controller. However, this architecture has a drawback on the RPi, that it involves separate processes and unnecessarily-onerous message-passing of raw frame data. In the RPi case we can just circumvent that - effectively we make sure that the message-passing interface just feeds through to a more direct functional API layer that the RPi python Sync Controller can call through to directly. Even if we do that, the RPi code will need a moderate amount of refactoring to separate out e.g. pin access from the sync code (we would have an API that *returns* predicted trigger times, rather than a function call that makes the prediction *and* messes with the pins if required), but that is a good thing to be doing anyway.

------------------------------------ **from here down these are just notes to myself; work in progress ------------------------------------**

**A brain-dump which felt relevant**, but which currently doesn't make any particular comment on the open-optical-gating architecture...

In my ObjC code, I have two separate entities, the SyncController and the SyncAnalyzer. The former is tightly integrated into the GUI, runs in the "main event loop", and is responsible for holding the "settings" object. The latter is completely stateless, runs asynchronously in response to frame arrivals, is multithreaded, and could in principle be processing a significant backlog of frames and therefore be "lagging behind" the GUI. It is passed frame objects, to which are attached the "settings" object *as it was**at the time the frame was received*. I can't now remember exactly why I felt that was necessary and/or a good idea, but it must have had something to do with the backlog scenario. Off the top of my head, I now can't think what specific and significant problem that might have been addressing, beyond a general concern that changes to the sync parameters via the GUI would be having a "retrospective" effect on frames that had been received before the user changed the setting. Clearly we don't want to get into a situation where we are backlogged, anyway, because the sync will not work well under those circumstances. [To be honest, if we have more than a tiny backlog when generating triggers, we would be better off just abandoning most of the backlogged frames... although then we will need to be careful to allow for variable time gaps in the phase history]

-> evidently the pi analyze() method must return before the next frame is acquired. It's not clear what happens if it doesn't! Significant risk it won't, if doing LTU. What happens then...?

It would make a lot more sense, and save a lot of passing around of long parameter lists, if there was one object that maintains the pog settings, history, *and* does the analysis logic. Or at a minimum, the history buffers required by LTU could be maintained by that LTU object (if that seems appropriate).

determine_reference_period passes the "settings" object back and forth, *and modifies it*. I need to think about the best way to handle this. It gets confusing (and DRP does not belong as methods inside a "settings" class, so we can't resolve the problem that way). Actually, the modification is trivial and probably can just be a return value instead. In general, I think we may be able to resolve by distinguishing between true "settings" parameters, and *state* (i.e. reference period and frames). If the state is maintained in some sort of sync object, we can pass settings back and forth as needed more easily, I think. The key thing, consistent with my own code, is that there is only one object responsible for curating *and modifying* that sync state.

**Architectural considerations**

- RPi camera object, remote camera via message-passing
- Separate analyzer object
- Settings object [who owns?]
- Pins
- Status reporting via messages
- Stages [currently they interact with stages in pog_state() and analyze() (!)]
- Emulation


**Chas writes**

My general rule of thumb for the different log levels is:

- trace for 'stuff that I may need to see but is impossibly to print cleanly,' e.g. large arrays that would just fill up my terminal
- debug for 'stuff I will want to print out if something breaks`
- info for 'expected breaks with no consequences`, in the sense that the code is working (so it's a success) but whats working is that something isn't working (so not a success), e.g. logger.info('No object found but I can carry on find so this is just info.')
- success for 'this worked and most users will want to know about it', these are sort of things that non-devs will want to know, e.g. 'i connected to the stages fine'
- warning for 'expected breaks with possible consequences', so like info except this could cause bad results or a later failure
- error for unexpected breaks, so when something the user requests happen can't, e.g. the user says move the stage but the code can't
- critical for failure, e.g. the user says 'get me a stage controller' and the code can't - this will break all future requests to do with the stage so it is critical and not just an error.

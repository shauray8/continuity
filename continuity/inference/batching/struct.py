import torch 

class Request:
    def __init__(self, user_id, prompt, total_steps, current_step=0):
        self.request_id = request_id
        self.prompt = prompt
        self.total_steps = total_steps
        self.current_step = current_step
        self.latents = None 
        self.completed = False

class ContinuousBatcher:
    def __init__(self, max_batch_size, model):
        self.queue = []
        self.max_batch_size = max_batch_size
        self.model = model
        self.batch = []

    def add_request(self, request):
        self.queue.append(request)

    def form_batch(self):
        self.batch = []
        for request in self.queue:
            if len(self.batch) < self.max_batch_size and not request.completed:
                self.batch.append(request)

    def process_batch(self):
        latents_list = [req.latents for req in self.batch]
        steps_list = [req.current_step for req in self.batch]

        ## add custom FLUX implimentation for this [keeping format similar to HF]
        new_latents = self.model.denoise(latents_list, steps_list)

        for i, request in enumerate(self.batch):
            request.latents = new_latents[i]
            request.current_step += 1

            if request.current_step >= request.total_steps:
                request.completed = True

    def remove_completed_requests(self):
        compeleted_batches = []
        compeleted_batches = [req for req in self.queue if req.completed] 
        self.queue = self.queue - compeleted_batches
        return compeleted_batches


## tbd : finding a better way of pipelining request to here
def check_for_new_requests():
    return [Request(request_id, prompt, total_steps=50) for user_id, prompt in get_user_input()]

def initialize_latents(height, width, pipeline):
    dummy_vector=torch.zeros(1,4,height//pipeline.vae_scale_factor,width//pipeline.vae_scale_factor)
    return torch.randn_like(dum, device="cuda", dtype=torch.float16)

def main():
    batcher = ContinuousBatcher(max_batch_size=4, model=FLUX_MODEL)
    completed_requests = []
    while True:
        incoming_requests = check_for_new_requests()

        for req in incoming_requests:
            latents = initialize_latents(req.height, req.width)
            req.latents = latents
            batcher.add_request(req)

        batcher.form_batch()

        if len(batcher.batch) > 0:
            batcher.process_batch()

        completed_requests = batcher.remove_completed_requests()

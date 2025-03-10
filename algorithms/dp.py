import math
import logging
import numpy as np
from algorithms.base import SchedulingAlgorithm

class Dp(SchedulingAlgorithm):
    def __init__(self, allocation_window, beta, logging_level, starting_allocation=None,
                 static=None, profiling_mode=False,accelerator_type="GPU_PASCAL"):  #undo 在simulator类中加入属性accelerator_type 表明同构的计算资源类型
        SchedulingAlgorithm.__init__(self, 'ILP')

        self.log = logging.getLogger(__name__)
        self.log.addHandler(logging.FileHandler('logs/dp/output.log', mode='w'))
        self.log.setLevel(logging_level)

        self.allocation_window = allocation_window

        self.initial_beta = beta
        self.beta = beta

        self.simulator = None
        self.num_isi = 0
        
        # We cache the latest solution
        self.cached_solution = None

        self.starting_allocation = starting_allocation
        self.static = static

        self.profiling_mode = profiling_mode

        self.accelerator_type=accelerator_type


    def is_simulator_set(self):
        if self.simulator is None:
            return False
        else:
            return True
        
    def set_simulator(self, simulator):
        self.simulator = simulator

        if self.starting_allocation is not None:
            predictor_dict, canary_dict, dp_x = self.get_solution_from_file(self.starting_allocation)
            self.log.info(f'ilp_x: {dp_x}')
            self.log.info(f'canary_dict: {canary_dict}')
            self.log.info(f'predictor_dict: {predictor_dict}')
            #self.simulator.apply_ilp_solution(predictor_dict, canary_dict, ilp_x)            undo 将dp方案在simulator上实施

    def run(self, observation, num_acc_types, num_max_acc):
        num_isi = observation.shape[0] - 1
        self.num_isi = num_isi
        
        current_alloc = observation[0:num_isi, 0:num_acc_types]

        # EWMA over sliding window
        demand_since_last = self.simulator.ewma_demand.ravel()  #得到dp的输入，吞吐量的要求
        # divide demand by time elapsed since last measurement to get demand in
        # units of requests per second
        demand = demand_since_last / (self.allocation_window / 1000)
        demand = np.ceil(demand)  #吞吐量由于要是整数，所以向上取整
        self.log.info(f'demand: {sum(demand)}')

        missed_requests = observation[0:num_isi, -1]

        largest_batch_sizes = self.simulator.get_largest_batch_sizes()  #获取(accelerator_type, model_variant)决定的最大batch_size信息
        
        profiled_latencies = self.simulator.model_variant_runtimes   #为后续算吞吐量做准备
        accelerator_type=self.accelerator_type
        acc_latencies = {}
        if accelerator_type == 'CPU':
            acc_latencies = profiled_latencies[1]
        elif accelerator_type == 'GPU_AMPERE':
            acc_latencies = profiled_latencies[2]
        elif accelerator_type == 'VPU':
            acc_latencies = profiled_latencies[3]
        elif accelerator_type == 'GPU_PASCAL':
            acc_latencies = profiled_latencies[4]

        all_models=[]  #预处理所有任务的模型变种的集合，all_models[isi]代表第isi类任务的模型变种
        for isi in range(num_isi):
            all_models[isi]={}
            isi_name = self.simulator.idx_to_executor[isi]
            model_variants = self.simulator.model_variants[isi_name]

            for model_variant in model_variants:
                #获得不同加速器和对应模型变种的峰值吞吐量
                largest_batch_size = largest_batch_sizes[(accelerator_type, model_variant)]
                if largest_batch_size == 0:
                    latency = None
                else:
                    latency = acc_latencies[(isi_name, model_variant, largest_batch_size)]
                throughput= 0 if latency is None else largest_batch_size * 1000 / latency
                throughput=math.floor(throughput)  #吞吐量向下取整
                #获得模型变种的准确率
                acc = self.simulator.model_variant_accuracies[(isi_name, model_variant)]    

                all_models[isi][model_variant]=[throughput,acc]

        gpu_num=num_max_acc

        least_num=self.least_gpu(all_models,demand)  #计算出每个任务需要的最少gpu数量，然后将多余的进行分配

        remain_num=gpu_num-sum(least_num)
        extra_num=np.zeros(num_isi)
        for isi in range(0,num_isi-1):
            extra_num[isi]=remain_num*(float(demand[isi])/sum(demand))  #最朴素的做法，按照demand多少进行分配      undo后续有时间进行改进
        if num_isi>1:
            extra_num[num_isi-1]=remain_num-sum(extra_num) #最后一个任务需要拿掉所有剩下的

        used_num=least_num+extra_num  #确定下了每个任务的gpu数量情况 后面直接使用背包算法求有限情况下的最大准确率策略

        required_predictors={}
        canary_dict=[]
        for isi in range(num_isi):
            solution=self.sub_problem(isi,demand,used_num[isi],all_models)
            #记录所有需要的acc_model数量
            required_predictors=required_predictors | solution
            #计算isi指定任务的路由百分比
            canary_dict[isi]={}
            throughput_sum=0
            for (model_variant,accelerator_type) in solution:
                throughput_sum+=all_models[isi][model_variant][0]*solution[(model_variant,accelerator_type)]  #计算指定isi任务的所有模型的吞吐量
            for(model_variant,accelerator_type) in solution:
                percentage=all_models[isi][model_variant][0]*solution[(model_variant,accelerator_type)]/float(throughput_sum)  #计算不同(model,acc)组合的吞吐量占比形成路由信息
                canary_dict[isi][(model_variant,accelerator_type)]=percentage

        return required_predictors,canary_dict
    
    def least_gpu(self,all_models,demand):
        cnt=[]  #记录所有任务所需的最小gpu数量，按isi标号
        for isi in range(self.num_isi):
            max_t=0 #该任务下所有模型中最大吞吐量
            for model_variant in all_models[isi]:
                max_t=max(max_t,all_models[isi][model_variant][0])
            tmp=math.ceil(demand[isi]/float(max_t))  #tmp代表需要最烂的模型副本的数量
            cnt[isi]=tmp #undo 后续需要修改，由于一个模型可能需要不止一个gpu，这里tmp需要乘以gpy使用数
        return cnt

    def sub_problem(self,isi,demand,gpu_num,all_models):  #undo 有关demand 由于使用背包算法，需要将demand增大一部分再进行求解
        current_models=all_models[isi]
        model_num=len(current_models)
        model_name=list(current_models.keys())
        
        mul=np.zeros((demand+1, gpu_num+1))    #动态规划数组   时间复杂度是三维背包,空间复杂度优化为二维
        mul[:] = -1 #所有元素都设置为-1，除了mul[0][0]
        for i in range(0,gpu_num+1):
            mul[0][i]=0     #demand=0时，gpu_num不论为多少，mul[0][i]=0都是合理的
        record=np.zeros((demand+1, gpu_num+1))  #用于复原模型选择过程，reocrd[j][k]代表在吞吐量严格为j，gpu数量不大于k的情况下，mul[j][k]最大使用的是哪一个模型

        for i in  range(1,model_num+1):
            for j in range(1,demand+1):
                for k in range(1,gpu_num):
                    (t,a)=current_models[model_name[i-1]]  # t->throughput   a->accuracy
                    if j>=t and k>=1: #undo  这里的1后续要改成模型使用的gpu数量
                        solve1= -1 if mul[j-t][k] == -1 else mul[j-t][k-1]+t*a #对应使用第i个模型的情况    undo 这里[k-1]只是目前模型参数少，每个模型仅使用一个gpu，后续需要修改成大模型时，需要加上模型的gpu占用数量
                    else:
                        solve1=-1
                    solve2=mul[j][k] #对应当前不使用第i个模型的情况
                    if solve1>solve2:  #当在给定条件下，使用第i个模型的时候，导致一个新的最大值，就记录下所使用的模型
                        record[j][k]=i
                    mul[j][k]=max(solve1,solve2)

        max_mul=mul[demand][gpu_num] #undo 后续改进，demand可以超出一部分，所有可能的demand，对gpu_num这一维度进行遍历，因为demand要固定，但是gpu_num这一维度不需要，只要求不大于
        tmp_demand=demand
        tmp_gpu_num=gpu_num
        cnt=np.zeros(model_num)  #记录调度策略，每个模型有多少副本
        while max_mul:
            tmp=record[tmp_demand][tmp_gpu_num]-1   #由于在记录的时候record[j][k]是从1开始
            cnt[tmp]+=1    
            (sub_demand,sub_gpu_num)=current_models[model_name[tmp]]
            tmp_demand-=sub_demand
            tmp_gpu_num-=tmp_gpu_num

        #将cnt转换为required_predictors
        required_predictors={}
        for i in range(model_num):
            if cnt[i]:  #只有非0个数的模型才需要记录下来
                tuple_key=(current_models[i],self.accelerator_type)
                required_predictors[tuple]=cnt[i]
        return required_predictors      #后续可能需要返回一个最大准确率，返回两个变量(required_predictors,acc_max)
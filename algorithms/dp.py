import time
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
            self.simulator.apply_ilp_solution(predictor_dict, canary_dict, dp_x)            #undo 将dp方案在simulator上实施   这里照抄dp方案

    def get_solution_from_file(self, file):
        with open(file, mode='r') as rf:
            lines = rf.readlines()
            required_predictors = eval(lines[1].rstrip('\n'))
            canary_dict = eval(lines[3].rstrip('\n'))
            dp_x = None
            if len(lines) > 4:
                dp_x = eval(lines[5].rstrip('\n'))

            return required_predictors, canary_dict, dp_x

    def run(self, observation, num_acc_types, num_max_acc):
        num_isi = observation.shape[0] - 1
        self.num_isi = num_isi
        
        current_alloc = observation[0:num_isi, 0:num_acc_types]

        precision=1 #控制小数点后精度
        # EWMA over sliding window
        demand_since_last = self.simulator.ewma_demand.ravel()  #得到dp的输入，吞吐量的要求
        # divide demand by time elapsed since last measurement to get demand in
        # units of requests per second
        demand = demand_since_last / (self.allocation_window / 1000)*precision  #是浮点数，按照可以接受的精度进行扩展
        self.log.info(f'demand: {sum(demand)}')
        if sum(demand) == 0:
            self.log.error('No requests received, terminating DP.')
            return None

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
            isi_models={}
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
                throughput=math.floor(throughput*precision)  #按照可以接受的精度进行扩展，吞吐量向下取整
                #获得模型变种的准确率
                acc = self.simulator.model_variant_accuracies[(isi_name, model_variant)]    

                isi_models[model_variant]=[throughput,acc]
            all_models.append(isi_models)

        gpu_num=num_max_acc

        least_num=self.least_gpu(all_models,demand)  #计算出每个任务需要的最少gpu数量，然后将多余的进行分配

        remain_num=gpu_num-sum(least_num)
        extra_num=np.zeros(num_isi)
        for isi in range(0,num_isi):
            extra_num[isi]=int(math.floor(remain_num*(float(demand[isi])/sum(demand))))  #最朴素的做法，按照demand多少进行分配 最后剩余的gpu全部给demand最大的任务     undo后续有时间进行改进
        tmp_isi=0  #寻找最大demand的任务标号
        for isi in range(1,num_isi): 
            tmp_isi=isi if demand[isi]>demand[tmp_isi] else tmp_isi
        extra_num[tmp_isi]+=remain_num-sum(extra_num) 

        used_num=least_num+extra_num  #确定下了每个任务的gpu数量情况 后面直接使用背包算法求有限情况下的最大准确率策略

        required_predictors={}
        canary_dict=[]
        for isi in range(num_isi):
            solution=self.sub_problem(isi,int(demand[isi]),int(used_num[isi]),all_models)
            #记录所有需要的acc_model数量
            required_predictors=required_predictors | solution
            #计算isi指定任务的路由百分比
            tmp_canary_dict={}
            throughput_sum=0
            for (model_variant,accelerator_type) in solution:
                throughput_sum+=all_models[isi][model_variant][0]*solution[(model_variant,accelerator_type)]  #计算指定isi任务的所有模型的吞吐量
            for(model_variant,accelerator_type) in solution:
                percentage=all_models[isi][model_variant][0]*solution[(model_variant,accelerator_type)]/float(throughput_sum)  #计算不同(model,acc)组合的吞吐量占比形成路由信息
                tmp_canary_dict[(model_variant,accelerator_type)]=percentage
            canary_dict.append(tmp_canary_dict)

        self.simulator.apply_dp_solution(required_predictors,canary_dict)#直接将调度策略应用在simulator上
        return
    
    def least_gpu(self,all_models,demand):
        cnt=[]  #记录所有任务所需的最小gpu数量，按isi标号
        for isi in range(self.num_isi):
            max_t=0 #该任务下所有模型中最大吞吐量
            for model_variant in all_models[isi]:
                max_t=max(max_t,all_models[isi][model_variant][0])
            tmp=math.ceil(demand[isi]/float(max_t))  #tmp代表需要最烂的模型副本的数量
            cnt.append(tmp)#undo 后续需要修改，由于一个模型可能需要不止一个gpu，这里tmp需要乘以gpu使用数
        return cnt

    def sub_problem(self,isi,target_demand,gpu_num,all_models):  #target_demand是系统所需的请求量，经过条件判断后，demand是扩增后的需求量
        current_models=all_models[isi]
        alpha=1.2#扩增系数

        #确定demand
        min_throughput=0
        for value in current_models.values():
            if value[0]>0:
                min_throughput=min(min_throughput,value[0]) if min_throughput>0 else value[0]
        if target_demand<min_throughput:
            demand=min_throughput  #demand超级小的情况
        else:
            demand=math.floor(target_demand*alpha) 
        model_num=len(current_models)
        model_name=list(current_models.keys())
        
        #开始背包算法
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
        
        #在背包计算完后进行选择最佳策略  选择record_demand  record_num
        solutions=[]  #先将所有可能的demand情况下的方案记录下来，根据准确率从大到小排序  前list_len个准确率中找吞吐量最大的
        list_len=5
        for d in range(target_demand,demand+1):
            max_mul=0
            tmp_record_num=0
            for n in range(1,gpu_num+1):
                if(mul[d][n]>max_mul):
                    max_mul=mul[d][n]
                    tmp_record_num=n
            solutions.append([max_mul/d,d,tmp_record_num])  #准确率 需求 gpu数量
        solutions.sort(key=lambda x: x[0], reverse=True) #按准确率从大到小排序
        max_demand=0
        record_demand=0        
        record_num=0
        for i in range(0,min(list_len,len(solutions))): #准确率前list_len中找到吞吐量最大的
            if(solutions[i][1]>max_demand):
                max_demand=solutions[i][1]
                (record_demand,record_num)=solutions[i][1:]
            

        #获取最佳策略后，根据dp反推每个模型的数量 记录在cnt中
        max_mul=mul[record_demand][record_num] #do 后续改进，demand可以超出一部分，所有可能的demand，对gpu_num这一维度进行遍历，因为demand要固定，但是gpu_num这一维度不需要，只要求不大于
        tmp_demand=record_demand
        tmp_gpu_num=record_num
        cnt=np.zeros(model_num)  #记录调度策略，每个模型有多少副本
        while max_mul>0:
            tmp=int(record[tmp_demand][tmp_gpu_num]-1)   #由于在记录的时候record[j][k]是从1开始
            cnt[tmp]+=1    #模型数量加1
            (sub_throughput,sub_acc)=current_models[model_name[tmp]]
            tmp_demand-=sub_throughput
            tmp_gpu_num-=1  #undo 后续需要改为实际使用gpu数量
            max_mul-=sub_throughput*sub_acc
            
        #sum(cnt)可能并不等于gpu_num，因为不一定所有的gpu都被使用到了，剩余的gpu全部加载最优的模型
        record_i=-1 #找出最优模型，需要准确率最高同时满足吞吐量不为0
        for i in range(0,model_num):
            (t,a)=current_models[model_name[i]]
            if(t>0):
                if record_i==-1:
                    record_i=i
                elif a>current_models[model_name[record_i]][1]:
                    record_i=i
        cnt[record_i]+=gpu_num-sum(cnt)

        #转换输出格式  将cnt转换为required_predictors
        required_predictors={}
        for i in range(model_num):
            if cnt[i]>0:  #只有非0个数的模型才需要记录下来
                tuple_key=(model_name[i],self.accelerator_type)
                required_predictors[tuple_key]=cnt[i]
        return required_predictors      #后续可能需要返回一个最大准确率，返回两个变量(required_predictors,acc_max)
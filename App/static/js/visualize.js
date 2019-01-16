 /* TODO
 * 1.渐进渲染progressive
 * 2.定制dataView,跳转到table页或者将table显示在下方……
 * 3.对聚类情况设置图例，颜色对应聚类后的group
 * 4.图像数量超出现有，设置翻页或者加长页面
 */


 /* 单一批次数据绘制三维三点图
  * param dataObj: 数据对象，拥有三个属性head,body,labels(optional)
  *     head 表示三个轴的名称
  *     body 是数据点坐标
  *     labels 是聚类后的类标号
  */
function draw3D (data,labels){
    echarts.dispose(document.getElementById('user_graph'))
    $('#user_graph').css({'width':'100%'})
    var myChart = echarts.init(document.getElementById('user_graph'));

    option = {
        title:{
            text: '测试员行为模型',
            subtext: '可用鼠标调整观察角度和大小',
            left: 'center'
        },
        toolbox: {
            show: true,
            feature: {
                saveAsImage: {}
            }
        },
        xAxis3D: {
            type: 'value',
            min: 0,
            max: 1
        },
        yAxis3D: {
            type: 'value',
            min: 0,
            max: 1
        },
        zAxis3D: {
            type: 'value',
            min: 0,
            max: 1
        },
        grid3D: {
            boxWidth: 100,
            boxDepth: 80,
            light: {
                main: {
                    intensity: 1.2
                },
                ambient: {
                    intensity: 0.3
                }
            }
        },tooltip:{
            show:true,
            formatter: function(params,ticket,callback){
                return '<p><b>测试员：</b>'+params.dataIndex+'号</p>'+
                '<p><b>伪造数据频率：</b>'+params.data[0].toFixed(2)+'</p>'+
                '<p><b>误输入频率：</b>'+params.data[1].toFixed(2)+'</p>'+
                '<p><b>选错量程频率：</b>'+params.data[2].toFixed(2)+'</p>'
            }
        },
        series: [{
            type: 'scatter3D',
            data: data,
            emphasis:{
                itemStyle:{color:'#000'}
            },
            itemStyle: {
                normal:{
                    color: function(params){
                        var dataIndex = params.dataIndex;
                        var colors=['#c23531','#2f4554', '#61a0a8', '#d48265', '#91c7ae','#749f83',  '#ca8622', '#bda29a','#6e7074', '#546570', '#c4ccd3']
                        var label = labels[dataIndex];
                        return colors[label];
                    }
                }

            }
        }]
    }
    myChart.setOption(option)
}

/* 单一批次数据绘制二维散点图
 * param dataObj:数据对象，拥有三个属性mistake_rate,line_scores,labels
 *      mistake_rate 表示本批次数据误输率
 *      samples_x，samples_y 表示观测值舒适度得分分布
 *      normal_x，normal_y 表示常规模型舒适度得分分布
 */
function draw_mistake_rate(dataObj){
    echarts.dispose(document.getElementById('mistake_scatter'))
    var myChart = echarts.init(document.getElementById('mistake_scatter'));
    var data = []
    var normal = []
    for(var i =0;i<dataObj.samples_x.length;i++){
        data.push([dataObj.samples_x[i],dataObj.samples_y[i]])
    }
    for(var j = 0; j<dataObj.normal_x.length; j++){
        normal.push([dataObj.normal_x[j],dataObj.normal_y[j]])
    }

    var option={
        title:{
            text: '误输率：'+dataObj.mistake_rate.toFixed(3)
        },
        grid: {
            width: '80%',
            height: '80%'
        },
        legend:{
            data:['观测值输入得分分布','输入得分常规模型分布']
        },
        xAxis: {
            type: 'value'
        },
        yAxis: {
            type: 'value'
        },
        tooltip:{
            trigger:'item',
            formatter:function(param){
                return '序号：'+(param.dataIndex+1)
            }
        },
        series: [{
            type:'scatter',
            name:'观测值输入得分分布',
            data: data,
            smooth:true,
            showSymbol:true,
        },{
            type:'scatter',
            name:'输入得分常规模型分布',
            data:normal,
            smooth:true,
            showSymbol:false
        }]
    }
    myChart.setOption(option);
}

/* 单一批次数据四组热力图
 * param dataObj:数据对象，拥有两个属性score_matrix,forge_rate
 *      forge_rate 表示本批次数据伪造率
 *      seg_num 分段数
 *      score_matrix 表示序列相似度得分矩阵
 */
 function drawHeatmap(dataObj){
    echarts.dispose(document.getElementById('forge_heatmap'))
    var myChart = echarts.init(document.getElementById('forge_heatmap'));
    var scores = [];
    var xData=[];
    var yData=[];
    var max=0;
    var seg_num = dataObj.seg_num;
    for (var matrix of dataObj.score_matrix){
        var quarter=[];
        var xq=[...Array(matrix.length).keys()]
        var yq=[...Array(matrix[0].length).keys()];
        for (var i=0;i<matrix.length;i++){
            var row = matrix[i];
            for(var j=0;j<row.length;j++){
                quarter.push([i,j,row[j]]);
                if (row[j]>max)
                    max=row[j];
            }
        }
        scores.push(quarter);
        xData.push(xq);
        yData.push(yq);
    }
    var grids=[];
    var xAxes=[];
    var yAxes=[];
    var serieses=[];
    for (var i=0;i<seg_num;i++){
        if(i==0){
            grids.push({right: '7%', bottom: '9%', width: '37%', height: '36%'})
        }else if(i==1){
            grids.push({left: '12%', bottom: '9%', width: '37%', height: '36%'})
        }else if(i%2==0){
            grids.push({right: '7%', top: '8%', width: '37%', height: '36%'})
        }else{
            grids.push({left: '12%', top: '8%', width: '37%', height: '36%'})
        }
        xAxes.push({gridIndex:i,type:'category',data:xData[i]});
        yAxes.push({gridIndex:i,type:'category',data:yData[i]});
        serieses.push({
            name:'序列相似度',
                type:'heatmap',
                xAxisIndex:i,
                yAxisIndex:i,
                data:scores[i],
                itemStyle: {
                    emphasis: {
                        borderColor: '#333',
                        borderWidth: 1
                    }
                },
                progressive: 1000,
                animation: false
        })
    }
    var option={
        title:{
            subtext:'伪造概率：'+dataObj.forge_rate.toFixed(3),
            left:'center'
        },
        tooltip:{
            formatter:function(param){
                var tooltip = param.marker+param.seriesName+':'+param.value[2];
                return tooltip;
            }
        },
        grid: grids,
        xAxis:xAxes,
        yAxis:yAxes, 
        visualMap: {
            min: 0,
            max: max,
            calculable: true,
            realtime: false,
            right:0,
            inRange: {
                color: ['#74add1', '#abd9e9',  '#fdae61', '#f46d43', '#d73027', '#a50026']
            }
        },
        series:serieses
    }
    myChart.setOption(option);
 }


 /* 单一批次数据极坐标图
  * param dataObj: 数据对象，拥有5个属性值 range_rate,theta,data,normal_theta,normal_data
  */
 function drawPolarmap(dataObj){
    echarts.dispose(document.getElementById('range_scatter'))
    var myChart = echarts.init(document.getElementById('range_scatter'));
    var option={
        title:{
            text: "量程误选择率分析"
        },
        legend:{
            data:['批次数据波动','正常波动'],
            left:'right'
        },
        polar:{},
        tooltip:{},
        angleAxis:{
            type:'value',
            boundaryGap:false,
            splitLine: {
                show: true,
                lineStyle: {
                    color: '#999',
                    type: 'dashed'
                }
            },
            axisLine: {
                show: false
            },
            axisLabel:{
                show:false
            }
        },
        radiusAxis:{
            type:'value',
            axisLine: {
                show: false
            },
            axisLabel: {
                show:false
            }
        },
        series:[{
            name:'批次数据波动',
            type:'line',
            data:dataObj.data,
            coordinateSystem:'polar'
        },{
            name:'正常波动',
            type:'line',
            data:dataObj.normal_data,
            coordinateSystem:'polar'
        }]
    }
    myChart.setOption(option);
 }
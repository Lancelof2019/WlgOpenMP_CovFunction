//#include "./buffertest.cpp"
struct Pixel{
int x;
int y;
int value;
Pixel(int cordx,int cordy,int pvalue){

  this->x=cordx;
  this->y=cordy;
  this->value=pvalue;
}
};


void bfs(int x,int y,int **matArr,int **distance,int rows,int cols){
queue<Pixel> q;
int dist=0;
bool **visr=new bool*[rows];
for(int i=0;i<rows;i++){
    visr[i]=new bool[cols];
    for(int j=0;j<cols;j++){
      visr[i][j]=false;
   }
}

Pixel p0=Pixel(x,y,dist);
q.push(p0);
visr[x][y]=true;
int temp;
temp=matArr[x][y];
int dx[4]={-1, 1, 0, 0};
int dy[4]={0, 0, -1, 1};
while(!q.empty()){
Pixel curp=q.front();
q.pop();
int curX=curp.x;
int curY=curp.y;
for(int i=0;i<4;i++){
 int nextX=curX+dx[i];
 int nextY=curY+dy[i];
 if(nextX < 0 || nextY < 0 || nextX> rows || nextY > cols ||matArr[nextX][nextY]>matArr[curX][curY]) {
   continue;
 }

 if(!visr[nextX][nextY]&&(matArr[nextX][nextY]<matArr[curX][curY])){
    visr[nextX][nextY]=true;
    curp.value++;
    distance[x][y]=curp.value;
    queue<Pixel>empq;
    swap(empq,q);
    break;
   }
  else if(matArr[nextX][nextY]==matArr[curX][curY]){
	  if(!visr[nextX][nextY]){
	   visr[nextX][nextY]=true;
           int a=0;
           Pixel next=Pixel(nextX,nextY,a);
           next.value=curp.value+1;
           q.push(next);
	  }
       }
   }
 }

}




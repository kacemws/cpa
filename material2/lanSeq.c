
/*
   Lanczos interpolation
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define TOL 1e-6
#define MAX(a,b) ((a)>=(b)?(a):(b))
#define MAX_SIZE 3000

typedef unsigned char Byte;

/* Read image from file. 
   It allocates and returns an array containing the image. The array will
   be large enough to contain either the image read or an image of size
   w2*h2 (whichever largest).
   On exit, w and h will be set to the width and height of the image read.
   If w2 is -1, it does not read the image, only stores its size into arguments
   w and h and returns NULL.

   If there is an error, it returns NULL and sets w and h to -1.
*/
Byte *read_ppm(char file[],int *w,int *h,int w2,int h2) {
  FILE *f;
  char type[10];
  Byte *A=NULL;
  int n;
  f=fopen(file,"rb");
  if (f==NULL) {
    *w=*h=-1;
    fprintf(stderr,"ERROR: Could not open \"%s\".\n",file);
  } else {
    fgets(type,sizeof(type),f);
    if (strcmp(type,"P6\n")) {
      fprintf(stderr,"ERROR: \"%s\" is not a PPM of type P6, but %s\n",
          file,type);
    } else {
      fscanf(f," #%*[^\n]"); /* skip possible comment */
      fscanf(f,"%d%d%*d%*c",w,h);
      if (w2!=-1) {
        n=(size_t)(*w)*(*h)*3;
        w2*=3*h2;
        A=(Byte*)malloc(MAX(n,w2)*sizeof(Byte));
        if (A==NULL) {
          fprintf(stderr,"ERROR: Not enough memory for %d bytes.\n",n);
        } else {
          fread(A,1,n,f);
        }
      }
    }
    fclose(f);
  }
  return A;
}

/* Write image to file */
void write_ppm(char file[],int w,int h,Byte *c) {
  FILE *f;
  f=fopen(file,"wb");
  if (f==NULL) {
    fprintf(stderr,"ERROR: Could not create \"%s\".\n",file);
  } else {
    fprintf(f,"P6\n%d %d\n255\n",w,h);
    fwrite(c,3,w*h,f);
    fclose(f);
  }
}

/* Lanczos kernel. It assumes -a <= x <= a */
double kernel(double x, int a) {
  double pi_x = M_PI*x;
  if ( pi_x < -TOL || pi_x > TOL ) 
    return a*sin(pi_x)*sin(pi_x/a)/(pi_x*pi_x);
  else 
    return 1;
}

/* Access element i of vector v, the elements of which are separated by
   a stride of inc */
#define vec_elem(v,i,inc) v[(i)*(inc)]


/* Given vector u of n elements, compute vector v of m elements, 
   using Lanczos interpolation with window size a.
   inc is the stride separating two consecutive elements of v. */
void resize1D( int n,double *u, int m,double *v,int inc, int a ) {
  int i,j,j1,j2;
  double s,e,x,xi,x0,fondo;
  e=(double)n/m; x0=e*0.5-0.5;
  for (i=0;i<m;i++) {
    xi=x0+e*i;
    j=(int)ceil(xi);
    j1=j-a; if (j1<0) {j1=0;fondo=u[0];} else fondo=0;
    j2=j+a; if (j2>n) {j2=n;fondo=u[n-1];}
    s=fondo;
    for (j=j1;j<j2;j++) {
      x = xi-j;
      s += (u[j]-fondo) * kernel(x,a);
    }
    vec_elem(v,i,inc) = s;
  }
}

/* Access a color component for pixel (x,y) of the 3-color matrix A (of width
   w), where each pixel is represented by three consecutive elements (R, G
   and B) */
#define cmat_elem(A,x,y,w) A[3*((x)+(w)*(y))]

/* Access element (x,y) of the matrix stored in A (w columns) */
#define mat_elem(A,x,y,w) A[(x)+(w)*(y)]

/* Given an image of width w and height h, resize one color component 
   (R, G or B) to width w2 and height h2.
   The color component is stored in the 3-color matrix A. A is overwritten
   with the resized image.
   B is an array of w*h2 elements, used internally as an auxiliar array.
   la is the window size to be used for the Lanczos method.
 */
void resize2D(int w,int h,Byte *A,int w2,int h2,double *B,int la) {
  int x,y;
  double v[MAX_SIZE];
  /* Resize height from A to B: A ( w * h ) -> B ( w * h2 ) */
  for (x=0;x<w;x++) {
    /* Copy column x of A to vector v, converting from Byte to double */
    for (y=0;y<h;y++) {
      v[y]=(double)cmat_elem(A,x,y,w);
    }
    /* Resize vector v, writing the result into column x of B */
    resize1D(h,v,h2,&mat_elem(B,x,0,w),w,la);
  }
  /* Resize width from B to A: B ( w * h2 ) -> A ( w2 * h2 ) */
  for (y=0;y<h2;y++) {
    /* Resize row y of B, writing the result into v */
    resize1D(w,&mat_elem(B,0,y,w),w2,v,1,la);
    /* Copy vector v to row y of A, converting from double to Byte */
    for (x=0;x<w2;x++) {
      cmat_elem(A,x,y,w2) = v[x]<0 ? 0 : v[x]>255 ? 255 : (Byte)v[x];
    }
  }
}

/* Given an image of width w and height h, resize it to width w2 and height h2.
   Array A contains the image and is overwritten with the resized image.
   la is the window size to be used for the Lanczos method.
 */
int resizeRGB(int w,int h,Byte *A,int w2,int h2,int la) {
  double *aux;
  aux = malloc(w*h2*sizeof(double));
  if (aux==NULL) {
    fprintf(stderr,"ERROR: Not enough memory.\n");
    return 1;
  }
  resize2D(w,h,A,w2,h2,aux,la);   /* red */
  resize2D(w,h,A+1,w2,h2,aux,la); /* green */
  resize2D(w,h,A+2,w2,h2,aux,la); /* blue */
  free(aux);
  return 0;
}

int main(int argc,char *argv[]) {
  int w,h,w2=0,h2=0,err,verbo=0,la=3;
  int t1 = 0, t2 = 0;
  double e=4;
  Byte *A;
  char *input="/labos/asignaturas/ETSINF/cpa/p2/Lenna256.ppm",*output="lanSeq.ppm",c,*arg;
  //char *prog = argv[0];
  while (argc>1 && argv[1][0]=='-') {
    switch(c=argv[1][1]) {
      case '-':output=NULL;break;
      case 'v':verbo=1;break;
      case 'w':
      case 'h':
      case 'e':
      case 'a':
        if (argv[1][2]=='\0') {
          arg=(++argv)[1];
          if (--argc==1) {
            fprintf(stderr,"WARNING: Number expected after option -%c.\n",c);
            break;
          }
        } else arg=&argv[1][2];
        switch(c) {
          case 'w':w2=atoi(arg);break;
          case 'h':h2=atoi(arg);break;
          case 'e': e=atof(arg);break;
          case 'a':la=atoi(arg);break;
        }
        break;
      default:fprintf(stderr,"WARNING: Unknown option -%c.\n",c);
    }
    ++argv;--argc;
  }
  if (argc>1) {input=*++argv;--argc;}
  if (argc>1) output=argv[1];
  read_ppm(input,&w,&h,-1,1);
  if (w==-1) return 1;
  if (w2>0||h2>0) {
    if (w2==0)w2=round(w*(double)h2/h);
    if (h2==0)h2=round(h*(double)w2/w);
  } else {
    w2=round(w*e); h2=round(h*e);
  }
  if (w2>MAX_SIZE || h>MAX_SIZE) {
    fprintf(stderr,"WARNING: Image is too large. Truncated to %d.\n",MAX_SIZE);
    if (w2>MAX_SIZE)w2=MAX_SIZE;
  }
  A=read_ppm(input,&w,&h,w2,h2);
  if (h>MAX_SIZE)h=MAX_SIZE;
  if (verbo)printf("%dx%d -> %dx%d\n",w,h,w2,h2);
  if (A!=NULL) {

    t1 = omp_get_wtime();
    err = resizeRGB(w,h,A,w2,h2,la);
    t2 = omp_get_wtime();

	printf("\n elapsed time : %d\n",t2-t1);

    if (err==0) {
	
      if (output!=NULL)write_ppm(output,w2,h2,A);
    }
    free(A);
  }
  return 0;
}


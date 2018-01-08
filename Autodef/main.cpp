#include <functional>

#include <Qt3DIncludes.h>
#include <GaussIncludes.h>
#include <FEMIncludes.h>

//Any extra things I need such as constraints
#include <ConstraintFixedPoint.h>
#include <ConstraintSlide.h>
#include <TimeStepperEulerImplicit.h>
#include <ForcePoint.h>

using namespace Gauss;
using namespace FEM;
using namespace ParticleSystem; //For Force Spring

/* Tetrahedral finite elements */

//typedef physical entities I need

//typedef scene
typedef PhysicalSystemFEM<double, NeohookeanHex> FEMLinearTets;

typedef World<double, std::tuple<FEMLinearTets *,PhysicalSystemParticleSingle<double> *>,
std::tuple<ForceSpringFEMParticle<double> *, ForcePoint<double> *>,
std::tuple<ConstraintSlide<double> *> > MyWorld;
typedef TimeStepperEulerImplicit<double, AssemblerParallel<double, AssemblerEigenSparseMatrix<double>>,
AssemblerParallel<double, AssemblerEigenVector<double> >> MyTimeStepper;

typedef Scene<MyWorld, MyTimeStepper> MyScene;


void preStepCallback(MyWorld &world) {
    // This is an example callback
}

void setMaterials(FEMLinearTets *test, double ySoft, double yStiff) {
    
    unsigned int nV = test->getImpl().getF().cols();
    unsigned int numStripes = 16;
    
    double yMax = test->getImpl().getV().col(0).maxCoeff();
    double yMin = test->getImpl().getV().col(0).minCoeff();
    
    double thickness = (yMax-yMin)/static_cast<double>(numStripes);
    
    //even strip = soft, odd strip = solid
    Eigen::Vector3d c;
    
    Eigen::MatrixXd &V = test->getImpl().getV();
    Eigen::MatrixXi &F = test->getImpl().getF();
    
    for(unsigned int iface=0;iface<test->getImpl().getF().rows(); ++iface) {
        
        c = V.row(F(iface,0));
        
        //centroid of element
        for(unsigned int ivert=1;ivert<nV; ++ivert) {
            c+=V.row(F(iface,ivert));
        }
        
        c /= static_cast<double>(nV);
    
        //check if in strip
        //unsigned int layerIndex = static_cast<unsigned int>(floor((c[0]- yMin)/thickness));
        //test->getImpl().getElement(iface)->setParameters(((layerIndex%2 == 0) ? ySoft : yStiff), 0.45);
        if(c(1) < 0.16) {
           test->getImpl().getElement(iface)->setParameters(ySoft, 0.45);
        } else {
            std::cout<<"LAYER\n";
           test->getImpl().getElement(iface)->setParameters(yStiff, 0.45);
        }
        
    }
}

int main(int argc, char **argv) {
    std::cout<<"Test Neohookean FEM \n";
    
    //Setup Physics
    MyWorld world;
    
    //new code -- load tetgen files
    //Eigen::MatrixXd V;
    //Eigen::MatrixXi F;
    
    //readTetgen(V, F, dataDir()+"/meshesTetgen/bucklingBarCoarse.node", dataDir()+"/meshesTetgen/bucklingBarCoarse.ele");

    //Load Geometry
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    
    
    //Voxel grid from libigl
    igl::grid(Eigen::RowVector3i(100, 10,10),  V);
    
    elementsFromGrid(Eigen::RowVector3i(100, 10, 10), V, F);

    V *= 2;
    
    FEMLinearTets *test = new FEMLinearTets(V,F);
    setMaterials(test, 5e5, 5e6);
    world.addSystem(test);
    fixDisplacementDirectionMin(world, test, 1, 1);
    fixDisplacementDirectionMin(world, test, 0, 0);
    fixDisplacementDirectionMin(world, test, 2, 0);
    fixDisplacementDirectionMax(world, test, 2, 0);
    fixDisplacementDirectionMax(world, test, 1, 0);
    
    
    
    //add squeezing forces
    //find all vertices with minimum x coordinate and fix DOF associated with them
    auto minX = test->getImpl().getV()(0,0);
    auto maxX = test->getImpl().getV()(0,1);
    std::vector<unsigned int> minV;
    std::vector<unsigned int> maxV;
    
    for(unsigned int ii=0; ii<test->getImpl().getV().rows(); ++ii) {
        
        if(test->getImpl().getV()(ii,0) < minX) {
            minX = test->getImpl().getV()(ii,0);
            minV.clear();
            minV.push_back(ii);
        } else if(fabs(test->getImpl().getV()(ii,0) - minX) < 1e-5) {
            minV.push_back(ii);
        }
        
        if(test->getImpl().getV()(ii,0) > maxX) {
            maxX = test->getImpl().getV()(ii,0);
            maxV.clear();
            maxV.push_back(ii);
        } else if(fabs(test->getImpl().getV()(ii,0) - maxX) < 1e-5) {
            maxV.push_back(ii);
        }
    }
    
    for(auto iV : maxV) {
        world.addForce(new ForcePoint<decltype(minX)>(&test->getQ()[iV], Eigen::Vector3d(-60,0,0)));
   }

    
    world.finalize(); //After this all we're ready to go (clean up the interface a bit later)
    
    auto q = mapStateEigen(world);
    q.setZero();
    
    MyTimeStepper stepper(0.002);
    
    //Display
    QGuiApplication app(argc, argv);
    
    MyScene *scene = new MyScene(&world, &stepper, preStepCallback);
    GAUSSVIEW(scene);
    
    return app.exec();
}
